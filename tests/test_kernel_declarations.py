"""Semantic boundaries for the first stage of kernel declaration migration."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
import torch

from torchcst import (
    Amplitude,
    AmpWidth,
    Biweight,
    Chart,
    DirectAmpWidth,
    Gaussian,
    Kernel,
    PolarAmpWidth,
    Separable,
    Triangle,
    Triweight,
    WendlandC2,
)
from torchcst.kernels import (
    AmpWidthSpec,
    DirectAmpWidthSpec,
    GaussianSpec,
    KernelSpec,
    NormalizationSpec,
    PolarAmpWidthSpec,
    ProfileBinding,
    TriweightSpec,
)


@pytest.mark.parametrize(
    "factory, identity, default_normalization",
    [
        (Gaussian, "gaussian", True),
        (Triweight, "triweight", True),
        (WendlandC2, "wendland_c2", True),
        (Triangle, "triangle", False),
        (Biweight, "biweight", False),
    ],
)
def test_shape_width_and_normalization_are_independent(
    factory, identity, default_normalization
):
    default = factory(0.2).declaration()
    raw = factory(0.7, normalize_columns=False).declaration()
    assert default.profile == raw.profile
    assert default.profile.id == identity
    assert default.parameterization.sigma != raw.parameterization.sigma
    assert (default.normalization.kind == "discrete_l2") == default_normalization
    assert raw.normalization == NormalizationSpec()
    if default_normalization:
        assert default.normalization.domain == "chart_sites"
        assert default.normalization.floor == (None if factory is Gaussian else 1e-6)
    with pytest.raises(FrozenInstanceError):
        default.profile.id = "changed"


def test_amp_width_derivative_law_is_declared_and_snapshot_is_independent():
    kernel = AmpWidth(
        sigma_min=0.2,
        sigma_max=1.0,
        profile=Triweight(0.2),
        law="inverse",
        couple_bandwidth=False,
    )
    spec = kernel.declaration()
    assert isinstance(spec.parameterization, AmpWidthSpec)
    assert spec.parameterization.law == "inverse"
    assert spec.parameterization.couple_bandwidth is False
    assert spec.composition == "separable"
    assert len(spec.profiles) == 2
    assert all(p.parameterization is None for p in spec.profiles)
    assert all(p.normalization.domain == "chart_sites" for p in spec.profiles)
    kernel.sigma_max.fill_(1.5)
    assert spec.parameterization.sigma_max == 1.0
    assert kernel.declaration().parameterization.sigma_max == 1.5
    json.dumps(asdict(spec))


@pytest.mark.parametrize(
    "factory, spec_type",
    [(DirectAmpWidth, DirectAmpWidthSpec), (PolarAmpWidth, PolarAmpWidthSpec)],
)
def test_activity_coordinates_are_distinct_from_execution_tuning(factory, spec_type):
    kernel = factory(
        amplitude_max=2.0,
        input_sigma_min=0.2,
        input_sigma_max=1.0,
        output_sigma_min=0.3,
        output_sigma_max=1.5,
        w_c=0.5,
    )
    spec = kernel.declaration()
    assert isinstance(spec.parameterization, spec_type)
    assert spec.parameterization.input_bounds.minimum == pytest.approx(0.2)
    assert spec.parameterization.output_bounds.minimum == pytest.approx(0.3)
    assert spec.initialization.id != spec.update.id
    if factory is DirectAmpWidth:
        kernel.site_chunk = 5
        kernel.atom_chunk = 2
        kernel.checkpoint_blocks = False
        assert kernel.declaration() == spec
    else:
        assert dict(spec.update.settings)["activity_mode"] == "finite_chord"
    json.dumps(asdict(spec))


def test_single_chart_requires_raw_profile_and_records_radial_composition():
    kernel = DirectAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.2,
        sigma_max=1.0,
        w_c=0.5,
        profile=Triweight(0.2, normalize_columns=False),
    )
    assert kernel.declaration(chart_count=1).composition == "radial"
    assert len(kernel.declaration(chart_count=1).profiles) == 1
    kernel.profile.normalize_columns = True
    with pytest.raises(ValueError, match="unnormalized"):
        kernel.declaration(chart_count=1)
    with pytest.raises(ValueError, match="chart_count"):
        kernel.declaration(chart_count=True)


def test_nested_amplitude_preserves_profile_order_and_widths():
    kernel = Amplitude(
        Separable(input_profile=Gaussian(0.2), output_profile=Triweight(0.4))
    )
    spec = kernel.declaration()
    assert spec.composition == "amplitude"
    assert spec.inner.profiles[0].profile == GaussianSpec()
    assert spec.inner.profiles[1].profile == TriweightSpec()
    assert spec.inner.profiles[1].parameterization.sigma == pytest.approx(0.4)


def test_snapshot_tracks_checkpoint_load_without_changing_checkpoint_keys():
    source = AmpWidth(sigma_min=0.2, sigma_max=1.5)
    target = AmpWidth(sigma_min=0.2, sigma_max=1.0)
    before = target.declaration()
    keys = set(target.state_dict())
    target.load_state_dict(source.state_dict())
    assert target.declaration() == source.declaration()
    assert before != target.declaration()
    assert set(target.state_dict()) == keys


def test_custom_profile_cannot_inherit_a_builtin_semantic_identity():
    class ShiftedTriweight(Triweight):
        def evaluate(self, chart, p):
            return super().evaluate(chart, p) + 1

    with pytest.raises(NotImplementedError, match="custom profiles"):
        ShiftedTriweight(0.2).declaration()


def test_custom_kernel_methods_remain_available_and_nonfactored_amplitude_rejects():
    class Custom(Kernel):
        def parameter_dim(self, *charts):
            return 1

        def initialize(self, *charts_and_atoms, mode):
            return torch.zeros(charts_and_atoms[-1], 1)

        def materialize_atoms(self, input_chart, output_chart, p):
            return p[:, None].expand(-1, output_chart.features, input_chart.features)

    kernel = Amplitude(Custom())
    chart = Chart.linspace(3, low=-1, high=1)
    p = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(
        kernel.materialize_atoms(chart, chart, p), torch.full((1, 3, 3), 6.0)
    )
    with pytest.raises(NotImplementedError, match="factorized backend"):
        kernel.factors(chart, chart, p)
    with pytest.raises(NotImplementedError, match="custom kernels"):
        kernel.declaration()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "none", "domain": "operator_sites"},
        {"kind": "discrete_l2"},
        {"kind": "discrete_l2", "domain": "chart_sites", "floor": 0},
        {"kind": "discrete_l2", "domain": "operator_sites", "floor": float("nan")},
    ],
)
def test_invalid_normalization_contracts_reject(kwargs):
    with pytest.raises(ValueError):
        NormalizationSpec(**kwargs)


def test_profile_width_required_without_kernel_parameterization():
    with pytest.raises(ValueError, match="bandwidth declaration"):
        KernelSpec(
            composition="radial", profiles=(ProfileBinding(profile=TriweightSpec()),)
        )
    with pytest.raises(ValueError, match="profile count"):
        KernelSpec(composition="separable", profiles=())
    with pytest.raises(TypeError):
        TriweightSpec(id="gaussian")


def test_declaration_import_and_snapshot_do_not_load_execution_modules():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", str(root / "src"))}
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from torchcst import AmpWidth, Triweight
from torchcst.kernels import TriweightSpec
kernel = AmpWidth(sigma_min=0.2, sigma_max=1.0, profile=Triweight(0.2))
kernel.declaration()
assert not any(name.startswith('torchcst._backends.torch.') for name in sys.modules)
assert 'triton' not in sys.modules
""",
        ],
        env=env,
        check=True,
    )


def test_execution_never_reads_a_declaration_snapshot(monkeypatch):
    kernel = AmpWidth(sigma_min=0.2, sigma_max=1.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration snapshots cannot enter execution")

    monkeypatch.setattr(Kernel, "declaration", forbidden)
    monkeypatch.setattr(type(kernel.profile), "declaration", forbidden)
    chart = Chart.linspace(3, low=-1, high=1)
    p = kernel.initialize(chart, chart, 2, mode="uniform").requires_grad_()
    kernel.materialize_atoms(chart, chart, p).square().sum().backward()
    assert torch.isfinite(p.grad).all()
