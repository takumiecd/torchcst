"""Pure declarations, common state, strict contracts and numerical execution."""

import copy
import io
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
import torch

import torchcst
from torchcst import BandwidthBounds, Chart, CSTLinear, CSTOptimizer, presets
from torchcst._backends.torch.kernels import execution as kernel_execution
from torchcst._backends.torch.profiles import execution as profile_execution
from torchcst.kernels import (
    AmpWidthSpec,
    BiweightSpec,
    DirectAmpWidthSpec,
    GaussianSpec,
    KernelSpec,
    NormalizationSpec,
    ParameterizationSpec,
    PolarAmpWidthSpec,
    ProfileBinding,
    ProfileSpec,
    StatePolicySpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)
from torchcst.kernels.options import KernelOptions
from torchcst.kernels.state import KernelState, ProfileState


@pytest.mark.parametrize(
    "shape", [GaussianSpec, BiweightSpec, TriangleSpec, TriweightSpec, WendlandC2Spec]
)
def test_shape_width_normalization_are_independent(shape):
    fixed = presets.fixed_profile(shape(), 0.2)
    raw = presets.fixed_profile(shape(), 0.7, normalize=False)
    assert fixed.profile == raw.profile
    assert fixed.parameterization.sigma != raw.parameterization.sigma
    assert fixed.normalization.domain == "chart_sites"
    assert raw.normalization == NormalizationSpec()
    json.dumps(asdict(fixed))
    with pytest.raises(FrozenInstanceError):
        fixed.profile.id = "changed"


def test_amp_width_snapshot_tracks_tensor_configuration():
    spec = presets.amplitude_width(
        sigma_min=0.2, sigma_max=1.0, law="inverse", couple_bandwidth=False
    )
    state = KernelState(spec)
    before = state.declaration()
    state.scalar("sigma_max").fill_(1.5)
    assert before.parameterization.sigma_max == 1.0
    assert state.declaration().parameterization.sigma_max == 1.5
    assert spec.parameterization.sigma_max == 1.0
    assert isinstance(spec.parameterization, AmpWidthSpec)
    assert all(p.parameterization is None for p in spec.profiles)


@pytest.mark.parametrize(
    "preset,coordinate",
    [
        (presets.direct_activity, DirectAmpWidthSpec),
        (presets.polar_activity, PolarAmpWidthSpec),
    ],
)
def test_activity_spec_is_independent_of_execution_options(preset, coordinate):
    bounds = BandwidthBounds(minimum=0.2, birth=0.4, maximum=1.0, upper_floor=0.2)
    spec = preset(amplitude_max=2.0, input_bounds=bounds, w_c=0.5)
    assert type(spec.parameterization) is coordinate
    first = KernelState(spec, options=KernelOptions(site_chunk=3, atom_chunk=2))
    second = KernelState(spec)
    assert first.declaration() == second.declaration()
    assert first.state_dict().keys() == second.state_dict().keys()
    assert tuple(first.parameters()) == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "none", "domain": "operator_sites"},
        {"kind": "discrete_l2"},
        {"kind": "discrete_l2", "domain": "chart_sites", "floor": 0},
        {"kind": "discrete_l2", "domain": "operator_sites", "floor": float("nan")},
    ],
)
def test_invalid_normalization_rejects(kwargs):
    with pytest.raises(ValueError):
        NormalizationSpec(**kwargs)


def test_width_is_declared_once():
    with pytest.raises(ValueError, match="bandwidth declaration"):
        KernelSpec(
            composition="radial", profiles=(ProfileBinding(profile=TriweightSpec()),)
        )
    spec = presets.amplitude_width(sigma_min=0.2, sigma_max=1.0)
    with pytest.raises(ValueError, match="once"):
        replace(spec, profiles=(presets.fixed_profile(GaussianSpec(), 0.2),) * 2)
    with pytest.raises(ValueError, match="profile count"):
        KernelSpec(composition="separable")


def test_pair_cannot_silently_use_operator_normalization():
    p = replace(
        presets.fixed_profile(TriweightSpec(), 0.2),
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=1e-6
        ),
    )
    with pytest.raises(ValueError, match="chart_sites"):
        presets.separable(input_profile=p, output_profile=p)


@pytest.mark.parametrize(
    "shape", [ProfileSpec(id="gaussian"), replace(GaussianSpec(), revision=2)]
)
def test_unknown_shape_or_revision_cannot_inherit_builtin_execution(shape):
    state = ProfileState(presets.fixed_profile(shape, 0.2))
    chart = Chart.linspace(3, spacing=0.2)
    with pytest.raises(ValueError, match="unsupported profile"):
        profile_execution.evaluate(state, chart, torch.zeros(1, 1))


@pytest.mark.parametrize(
    "change",
    [
        {"revision": 2},
        {"parameterization": ParameterizationSpec(id="amplitude_dependent_width")},
        {"initialization": StatePolicySpec(id="unknown")},
        {"update": StatePolicySpec(id="unknown")},
    ],
)
def test_backend_rejects_unrecognized_kernel_contract(change):
    spec = replace(presets.amplitude_width(sigma_min=0.2, sigma_max=1.0), **change)
    chart = Chart.linspace(3, spacing=0.2)
    with pytest.raises(ValueError, match="unsupported"):
        kernel_execution.initialize(KernelState(spec), chart, chart, 2)


def test_execution_never_reads_snapshots(monkeypatch):
    state = KernelState(presets.amplitude_width(sigma_min=0.2, sigma_max=1.0))

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot inside execution")

    monkeypatch.setattr(KernelState, "declaration", forbidden)
    monkeypatch.setattr(ProfileState, "declaration", forbidden)
    chart = Chart.linspace(3, spacing=0.2)
    p = kernel_execution.initialize(state, chart, chart, 2).requires_grad_()
    kernel_execution.materialize_atoms(state, chart, chart, p).square().sum().backward()
    assert torch.isfinite(p.grad).all()


def test_declaration_import_does_not_load_evaluators():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from torchcst import presets, TriweightSpec
p=presets.fixed_profile(TriweightSpec(),.2)
s=presets.separable(input_profile=p,output_profile=p)
assert not any(n.startswith('torchcst._backends.torch.') for n in sys.modules)
assert 'triton' not in sys.modules
""",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=True,
    )


def test_removed_public_classes_and_aliases_have_no_compatibility_entry():
    for name in (
        "Kernel",
        "Profile",
        "Gaussian",
        "Triweight",
        "Separable",
        "Amplitude",
        "AmpWidth",
        "AmplitudeBandwidthSeparable",
        "DirectAmpWidth",
        "PolarAmpWidth",
    ):
        assert not hasattr(torchcst, name)
    for name in (
        "base",
        "compact",
        "gaussian",
        "separable",
        "amplitude",
        "amplitude_bandwidth",
        "direct_amplitude_bandwidth",
        "polar_amplitude_bandwidth",
        "_declarations",
    ):
        assert not (
            Path(__file__).resolve().parents[1] / "src/torchcst/kernels" / f"{name}.py"
        ).exists()


def test_radial_amplitude_updates_and_checkpoint_are_weights_only_safe():
    from torchcst import LinePattern, ProductChart

    chart = ProductChart(
        shape=(3, 4), axes=(LinePattern(3, spacing=0.2), LinePattern(4, spacing=0.2))
    )
    spec = presets.amplitude(presets.radial(presets.fixed_profile(GaussianSpec(), 0.5)))
    model = CSTLinear(chart=chart, atoms=3, kernel=spec, dtype=torch.float64)
    optimizer = CSTOptimizer(torch.optim.AdamW(model.parameters()), model=model)
    model(torch.randn(2, 4, dtype=torch.float64)).square().sum().backward()
    optimizer.step()
    stream = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream
    )
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(saved["model"])
    other = CSTOptimizer(torch.optim.AdamW(restored.parameters()), model=restored)
    other.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(restored.dense_weight(), model.dense_weight())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude_max": 0.0},
        {"w_c": 0.0},
        {"kappa": 1.0},
        {"lower_kappa": 0.0},
        {"upper_decay_power": 0.0},
        {"upper_decay_power": 1.1},
        {"alpha_init": 1.1},
        {"radial_regularization": -1.0},
    ],
)
def test_activity_declaration_validation(kwargs):
    settings = {
        "amplitude_max": 2.0,
        "input_bounds": BandwidthBounds(
            minimum=0.1, birth=1.0, maximum=1.0, upper_floor=0.1
        ),
        "w_c": 0.5,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        presets.direct_activity(**settings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activity_gain": 0.0},
        {"activity_mode": "unknown"},
        {"dormant_expansion_rate": -1.0},
    ],
)
def test_polar_update_declaration_validation(kwargs):
    with pytest.raises(ValueError):
        presets.polar_activity(
            amplitude_max=2.0,
            input_bounds=BandwidthBounds(
                minimum=0.1, birth=1.0, maximum=1.0, upper_floor=0.1
            ),
            w_c=0.5,
            **kwargs,
        )


@pytest.mark.parametrize(
    "preset", [presets.amplitude_width, presets.direct_activity, presets.polar_activity]
)
def test_output_profile_has_its_own_declared_shape(preset):
    from torchcst.kernels.parameterizations import AmpWidthSpec

    spec = (
        preset(sigma_min=0.2, sigma_max=1.0)
        if preset is presets.amplitude_width
        else preset(
            amplitude_max=2.0,
            input_bounds=BandwidthBounds(
                minimum=0.2, birth=1.0, maximum=1.0, upper_floor=0.2
            ),
            w_c=0.5,
            **(
                {"composition": "separable"}
                if preset is presets.direct_activity
                else {}
            ),
        )
    )
    spec = replace(
        spec,
        profiles=(
            presets.profile(GaussianSpec(), normalize=False),
            presets.profile(TriweightSpec(), normalize=False),
        ),
    )
    state = KernelState(spec).double()
    chart = Chart.linspace(5, spacing=0.3).double()
    p = kernel_execution.initialize(state, chart, chart, 3).requires_grad_()
    input_factor, output_factor = kernel_execution.factors(state, chart, chart, p)
    if isinstance(spec.parameterization, AmpWidthSpec):
        precision = kernel_execution.coordinate(
            state, "bandwidth_precision", chart, chart, p
        )
        amplitude = p[:, 0]
        source, target = p[:, 1:2], p[:, 2:]
        pin, pout = precision, precision
    else:
        amplitude = kernel_execution.coordinate(state, "amplitude", chart, chart, p)
        sigmas = kernel_execution.coordinate(state, "bandwidth_sigmas", chart, chart, p)
        pin, pout = (s.reciprocal().square().detach() for s in sigmas)
        source, target = p[:, 2:3], p[:, 3:]
    expected_input = profile_execution.evaluate_with_precision(
        state.profiles[0], chart, source, pin
    )
    expected_output = (
        profile_execution.evaluate_with_precision(
            state.profiles[1], chart, target, pout
        )
        * amplitude[None]
    )
    torch.testing.assert_close(input_factor, expected_input)
    torch.testing.assert_close(output_factor, expected_output)
    actual_grad = torch.autograd.grad(
        (input_factor.sum() + output_factor.sum()), p, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad(
        (expected_input.sum() + expected_output.sum()), p
    )[0]
    torch.testing.assert_close(actual_grad, expected_grad)
