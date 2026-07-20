"""Continuous Gaussian CSTLinear numerical and mass contracts."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from torchcst.compute import CSTLinear
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


def _parts() -> tuple[SynapseStore, GaussianKernel, CSTLinear]:
    store = SynapseStore(
        "continuous",
        2,
        2,
        3,
        spec=RepresentationSpec.continuous(2, 2, bounds=(-1.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                "continuous",
                torch.tensor(
                    [[-0.6, 0.2], [0.1, 0.7], [0.8, -0.4]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [[-0.2, 0.5], [0.6, -0.7], [0.4, 0.9]],
                    dtype=torch.float64,
                ),
                torch.tensor([0.5, -1.25, 0.3], dtype=torch.float64),
                torch.arange(3, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        4,
        mu=torch.tensor(
            [[-1.0, -1.0], [-0.3, 0.2], [0.5, -0.4], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        initial_live=4,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        3,
        mu=torch.tensor(
            [[-0.8, 0.9], [0.0, 0.0], [0.9, -0.6]], dtype=torch.float64
        ),
        initial_live=3,
        dtype=torch.float64,
    )
    kernel = GaussianKernel(0.55).double()
    return store, kernel, CSTLinear(inputs, outputs, store, kernel)


def test_forward_and_w_s_t_sigma_gradients_match_dense_linear() -> None:
    store, kernel, module = _parts()
    assert isinstance(store.s, nn.Parameter) and isinstance(store.t, nn.Parameter)
    assert isinstance(kernel.sigma, nn.Parameter)
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(6, 3, dtype=torch.float64)

    factorized = module(x)
    dense = F.linear(x, module.dense_weight())
    torch.testing.assert_close(factorized, dense)

    factorized.backward(upstream, retain_graph=True)
    expected = {
        "w": store.w.grad.clone(),
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "sigma": kernel.sigma.grad.clone(),
        "x": x.grad.clone(),
    }
    for parameter in (store.w, store.s, store.t, kernel.sigma):
        parameter.grad = None
    x.grad = None
    dense.backward(upstream)

    for name, actual in (
        ("w", store.w.grad),
        ("s", store.s.grad),
        ("t", store.t.grad),
        ("sigma", kernel.sigma.grad),
        ("x", x.grad),
    ):
        torch.testing.assert_close(actual, expected[name])


def test_functional_mass_and_sigma_change_refresh_mass_scale() -> None:
    store, kernel, module = _parts()
    x = torch.randn(2, 4, dtype=torch.float64)

    module(x)
    view = store.view()
    k_in = kernel(module.in_neurons.mu, view.s)
    k_out = kernel(module.out_neurons.mu, view.t)
    expected = view.w.abs() * k_in.norm(dim=0) * k_out.norm(dim=0)
    torch.testing.assert_close(view.mass, expected)
    before = store.mass_scale.clone()

    with torch.no_grad():
        kernel.sigma.copy_(torch.tensor(0.25, dtype=torch.float64))
    module(x)
    refreshed = store.view()
    expected = (
        refreshed.w.abs()
        * kernel(module.in_neurons.mu, refreshed.s).norm(dim=0)
        * kernel(module.out_neurons.mu, refreshed.t).norm(dim=0)
    )
    torch.testing.assert_close(refreshed.mass, expected)
    assert not torch.equal(store.mass_scale, before)


def test_synapse_view_is_reacquired_only_after_structural_version_change() -> None:
    store, _, module = _parts()
    x = torch.randn(2, 4, dtype=torch.float64)
    original = store.view

    with patch.object(store, "view", wraps=original) as viewed:
        module(x)
        module(x)
        assert viewed.call_count == 1
        store.apply([SynapseDeath("continuous", original().ids[:1])])
        module(x)
        assert viewed.call_count == 2
