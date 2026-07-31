"""Independent CSTConv2d numerical and engine contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest
import torch
import torch.nn.functional as F

from torchcst.compute import CSTConv2d, conv2d_neuron_coordinates
from torchcst.engine import StructuralEngine
from torchcst.instruments import GradFieldEMA
from torchcst.policy import (
    EvenBudgetDistributor,
    InstrumentSpec,
    MagnitudeCourt,
    PeriodicCadence,
    QuotaRegime,
    SynapseLifecycle,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _parts() -> tuple[SynapseStore, GaussianKernel, CSTConv2d]:
    input_mu, output_mu = conv2d_neuron_coordinates(
        2, 3, (2, 3), channel_scale=0.8, spatial_scale=0.7, dtype=torch.float64
    )
    store = SynapseStore(
        "conv",
        3,
        1,
        5,
        spec=RepresentationSpec.continuous(3, 1),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                "conv",
                torch.tensor(
                    [
                        [0.1, 0.2, 0.3],
                        [0.7, 0.8, 0.2],
                        [0.5, 0.4, 0.9],
                        [0.3, 0.9, 0.6],
                        [0.9, 0.1, 0.5],
                    ],
                    dtype=torch.float64,
                ),
                torch.tensor([[0.1], [0.7], [0.4], [0.9], [0.55]], dtype=torch.float64),
                torch.tensor([0.4, -0.7, 0.2, 0.9, -0.3], dtype=torch.float64),
                torch.arange(5, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "conv-in",
        input_mu.shape[0],
        mu=input_mu,
        initial_live=input_mu.shape[0],
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "conv-out",
        output_mu.shape[0],
        mu=output_mu,
        initial_live=output_mu.shape[0],
        dtype=torch.float64,
    )
    kernel = GaussianKernel(0.35, learnable=True).double()
    conv = CSTConv2d(
        inputs,
        outputs,
        store,
        kernel,
        2,
        (2, 3),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(1, 2),
        bias=True,
    )
    with torch.no_grad():
        conv.bias.copy_(torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64))
    return store, kernel, conv


def test_conv_owns_continuous_map_without_linear_wrapper() -> None:
    store, kernel, conv = _parts()

    assert not hasattr(conv, "linear")
    assert conv.store is store
    assert conv.synapses is store
    assert conv.kernel_in is kernel
    assert conv.in_neurons.site == "conv-in"
    assert conv.out_neurons.site == "conv-out"


def test_coordinate_grid_matches_conv_flattening_and_box() -> None:
    inputs, outputs = conv2d_neuron_coordinates(2, 4, (2, 3))

    assert inputs.shape == (12, 3)
    assert outputs.shape == (4, 1)
    assert bool(((inputs >= 0.0) & (inputs <= 1.0)).all())
    assert bool(((outputs >= 0.0) & (outputs <= 1.0)).all())
    torch.testing.assert_close(inputs[0], torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(inputs[5], torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(inputs[6], torch.tensor([1.0, 0.0, 0.0]))


def test_forward_and_gradients_match_materialized_conv2d() -> None:
    store, kernel, conv = _parts()
    x = torch.randn(2, 2, 7, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, 4, 8, dtype=torch.float64)

    actual = conv(x)
    expected = F.conv2d(
        x,
        conv.dense_weight(),
        conv.bias,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
    )
    torch.testing.assert_close(actual, expected)

    actual.backward(upstream, retain_graph=True)
    factorized_grads = {
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "w": store.w.grad.clone(),
        "sigma": kernel.sigma.grad.clone(),
        "bias": conv.bias.grad.clone(),
        "x": x.grad.clone(),
    }
    for parameter in (store.s, store.t, store.w, kernel.sigma, conv.bias):
        parameter.grad = None
    x.grad = None
    expected.backward(upstream)

    for name, value in (
        ("s", store.s.grad),
        ("t", store.t.grad),
        ("w", store.w.grad),
        ("sigma", kernel.sigma.grad),
        ("bias", conv.bias.grad),
        ("x", x.grad),
    ):
        torch.testing.assert_close(value, factorized_grads[name])


def test_fast_materialized_path_matches_unfold_path_and_gradients() -> None:
    store, kernel, conv = _parts()
    with torch.no_grad():
        conv.in_neurons.gate[0] = 0.4
        conv.out_neurons.gate[1] = 0.25
    conv.implementation = "materialized"
    x = torch.randn(2, 2, 7, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, 4, 8, dtype=torch.float64)

    actual = conv(x)
    actual.backward(upstream, retain_graph=True)
    materialized_grads = {
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "w": store.w.grad.clone(),
        "sigma": kernel.sigma.grad.clone(),
        "bias": conv.bias.grad.clone(),
        "x": x.grad.clone(),
    }
    for parameter in (store.s, store.t, store.w, kernel.sigma, conv.bias):
        parameter.grad = None
    x.grad = None

    conv.implementation = "unfold"
    expected = conv(x)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected)
    for name, value in (
        ("s", store.s.grad),
        ("t", store.t.grad),
        ("w", store.w.grad),
        ("sigma", kernel.sigma.grad),
        ("bias", conv.bias.grad),
        ("x", x.grad),
    ):
        torch.testing.assert_close(value, materialized_grads[name])


@dataclass
class _ObserveLiveAtoms:
    requires: tuple[InstrumentSpec, ...] = field(
        default=(InstrumentSpec("grad_field", decay=0.0),), init=False
    )
    instruments: dict[str, GradFieldEMA] = field(default_factory=dict, init=False)

    def bind_instruments(self, site: str, instruments: Mapping[str, Any]) -> None:
        self.instruments[site] = instruments["grad_field"]

    def propose(self, view, budget, registry, rng):
        return ()


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_engine_capture_observes_unfolded_patch_rows(capture_mode: str) -> None:
    store, _, conv = _parts()
    proposer = _ObserveLiveAtoms()
    method = SynapseLifecycle(
        birth_factory=lambda lam: proposer,
        prune_factory=lambda: MagnitudeCourt(0.0),
        priceable=False,
        label="observe-live-atoms",
    )
    policy = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    stores = {
        store.site: store,
        conv.in_neurons.site: conv.in_neurons,
        conv.out_neurons.site: conv.out_neurons,
    }
    optimizer = torch.optim.SGD(conv.parameters(), lr=1.0e-3)
    engine = StructuralEngine(
        stores,
        policy,
        modules={store.site: conv},
        optimizer=optimizer,
        capture_mode=capture_mode,
    )

    x = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    engine.begin_update()
    optimizer.zero_grad()
    conv(x).square().mean().backward()
    engine.observe_microbatch()
    engine.finalize_backward()

    ids, scores = proposer.instruments[store.site].snapshot()
    torch.testing.assert_close(ids, store.view().ids)
    assert scores.shape == (5,)
    assert bool((scores > 0).any())


def test_isotropic_scales_equalize_every_chart_axis_spacing() -> None:
    """The scales must make one nearest-neighbour spacing serve all three
    input-chart axes -- that equality is the whole content of the fix, and
    the default (1.0, 1.0) violates it by (C-1)/2."""
    from torchcst.compute import conv2d_isotropic_scales

    for in_channels in (16, 32, 64):
        channel_scale, spatial_scale = conv2d_isotropic_scales(in_channels, 3)
        assert channel_scale == 1.0
        assert spatial_scale == pytest.approx(2.0 / (in_channels - 1))

        in_mu, _ = conv2d_neuron_coordinates(
            in_channels, 8, 3, channel_scale=channel_scale,
            spatial_scale=spatial_scale, dtype=torch.float64,
        )
        channel_axis = torch.unique(in_mu[:, 0])
        ky_axis = torch.unique(in_mu[:, 1])
        kx_axis = torch.unique(in_mu[:, 2])
        spacings = [
            float(axis.diff().min()) for axis in (channel_axis, ky_axis, kx_axis)
        ]
        assert spacings[0] == pytest.approx(spacings[1])
        assert spacings[0] == pytest.approx(spacings[2])
        # And it is still a chart inside the unit cube.
        assert float(in_mu.min()) >= 0.0 and float(in_mu.max()) <= 1.0

    # Fewer channels than taps: the compressed axis flips over.
    assert conv2d_isotropic_scales(2, 3) == (1.0, 1.0) or conv2d_isotropic_scales(2, 3)[0] < 1.0
    assert conv2d_isotropic_scales(1, 3) == (1.0, 1.0)


def test_the_default_chart_is_anisotropic_by_a_factor_that_grows_with_channels() -> None:
    """A regression pin on the defect itself: the default keeps tap spacing
    at 0.5 no matter how fine the channel axis gets, so one isotropic sigma
    cannot resolve both."""
    for in_channels, expected in ((16, 7.5), (32, 15.5), (64, 31.5)):
        in_mu, _ = conv2d_neuron_coordinates(in_channels, 8, 3, dtype=torch.float64)
        channel_spacing = float(torch.unique(in_mu[:, 0]).diff().min())
        tap_spacing = float(torch.unique(in_mu[:, 1]).diff().min())
        assert tap_spacing == pytest.approx(0.5)
        assert tap_spacing / channel_spacing == pytest.approx(expected)
