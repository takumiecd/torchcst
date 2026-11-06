"""Public initialization moments, independent sign oracle and failure atomicity."""

import copy
import itertools
import math

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    ChebyshevSpec,
    GaussianSpec,
    RadiusBounds,
    chart_presets,
    init,
    kernels,
)
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values


def layer(ni=16, no=32, atoms=128, dtype=torch.float64):
    return CSTLinear(
        chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
        atoms=atoms,
        dtype=dtype,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.3,
                input_bounds=RadiusBounds(
                    minimum=0.5, birth=1.0, maximum=3.0, upper_floor=1.0
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "scheme,expected", [("he_in", 64.0), ("he_out", 32.0), ("xavier", 2 * 16 * 32 / 48)]
)
def test_logical_rectangular_dense_target_and_actual_atom_energy(scheme, expected):
    m = layer()
    p = m.atoms.p
    before = p.detach().clone()
    kwargs = dict(q=0.0, generator=torch.Generator().manual_seed(87))
    if scheme.startswith("he"):
        report = init.kaiming_normal_(
            m,
            nonlinearity="relu",
            mode="fan_in" if scheme == "he_in" else "fan_out",
            **kwargs,
        )
    else:
        report = init.xavier_normal_(m, **kwargs)
    assert m.atoms.p is p and torch.equal(p[:, 2:], before[:, 2:])
    assert report.target_energy == pytest.approx(expected)
    assert report.relative_error <= 0.005 and 0 <= report.clipped_atoms <= 128
    # Explicit per-Atom matrices, rather than the initializer's factored norm
    # reduction. This verifies the conditional-sign operator energy target.
    matrices = m.materialized_atoms()
    energy = float(matrices.detach().square().sum())
    assert energy == pytest.approx(report.expected_energy, rel=1e-12, abs=1e-13)
    assert abs(energy / expected - 1) <= 0.005


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_q_encoding_repeatability_and_exact_center_preservation(q, dtype):
    m = layer(dtype=dtype)
    other = copy.deepcopy(m)
    centers = m.atoms.p[:, 2:].detach().clone()
    report = init.variance_scaling_(
        m, 0.0001, q=q, generator=torch.Generator().manual_seed(91)
    )
    again = init.variance_scaling_(
        other, 0.0001, q=q, generator=torch.Generator().manual_seed(91)
    )
    assert torch.equal(m.atoms.p, other.atoms.p) and report == again
    assert torch.equal(m.atoms.p[:, 2:], centers)
    polar = m.atoms.p[:, :2].double()
    actual = ((polar.square().sum(1) - 1) / 3).clamp(0, 1)
    torch.testing.assert_close(actual, torch.full_like(actual, q), atol=3e-7, rtol=0)
    a, ri, ro = decoded_values(m.kernel, m.atoms.p.double())
    assert (
        torch.isfinite(a).all()
        and torch.isfinite(ri).all()
        and torch.isfinite(ro).all()
    )
    assert m.atoms.p.grad is None


def test_overlapping_atom_sign_expectation_is_not_realized_weight_energy():
    m = layer(ni=8, no=8, atoms=3)
    with torch.no_grad():
        m.atoms.p[:, 2:] = m.atoms.p[0, 2:]
    report = init.variance_scaling_(
        m, 0.002, generator=torch.Generator().manual_seed(7)
    )
    table = m.atoms.p.detach().clone()
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        with torch.no_grad():
            m.atoms.p.copy_(table)
            m.atoms.p[:, 0] *= torch.tensor(signs, dtype=table.dtype)
        values.append(float(m.dense_weight().detach().square().sum()))
    assert sum(values) / 8 == pytest.approx(
        report.expected_energy, rel=1e-12, abs=1e-13
    )
    assert max(values) - min(values) > 1e-3


@pytest.mark.parametrize(
    "variance,q",
    [
        (0.0, 0.0),
        (math.nan, 0.0),
        (True, 0.0),
        (0.1, -0.1),
        (0.1, 1.1),
        (0.1, math.nan),
        (0.1, True),
        (1e20, 0.0),
    ],
)
def test_invalid_or_unreached_target_leaves_complete_state_unchanged(variance, q):
    m = layer(atoms=3)
    before = copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError):
        init.variance_scaling_(m, variance, q=q)
    for n, p in m.state_dict().items():
        if isinstance(p, torch.Tensor):
            assert torch.equal(p, before[n])
        else:
            assert p == before[n]


def test_unsupported_kernel_rejected_without_changing_parameters():
    grid = chart_presets.regular_grid(8, spacing=1.0, geometry="flat_torus")
    m = CSTLinear(
        grid,
        grid,
        atoms=8,
        kernel=kernels.product(
            kernels.l2(GaussianSpec()),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.3,
                bounds=BandwidthBounds(
                    minimum=0.1, birth=1.0, maximum=2.0, upper_floor=0.1
                ),
            ),
        ),
    )
    before = m.atoms.p.detach().clone()
    with pytest.raises(NotImplementedError):
        init.xavier_normal_(m)
    assert torch.equal(before, m.atoms.p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("scheme,q", [("he", 0.0), ("xavier", 0.5)])
def test_initialized_public_parameters_native_rectangular_full_vjp(scheme, q):
    from torchcst import FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        GLOBAL_ACTIVE_GRAD4_NATIVE,
    )

    model = layer(dtype=torch.float32)
    generator = torch.Generator().manual_seed(71)
    if scheme == "he":
        init.kaiming_normal_(model, nonlinearity="relu", q=q, generator=generator)
    else:
        init.xavier_normal_(model, q=q, generator=generator)
    reference = copy.deepcopy(model).double()
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(81))
    dy = torch.randn(4, 32, generator=torch.Generator().manual_seed(82))
    xx = x.double().requires_grad_()
    expected = reference(xx)
    expected_grad = torch.autograd.grad(expected, (xx, reference.atoms.p), dy.double())
    model = model.cuda()
    model.selector = FixedSelector(
        GLOBAL_ACTIVE_GRAD4_NATIVE, registry=make_algorithm_registry()
    )
    live = x.cuda().requires_grad_()
    actual = model(live)
    grad = torch.autograd.grad(actual, (live, model.atoms.p), dy.cuda())
    for got, want in zip((actual, *grad), (expected, *expected_grad), strict=True):
        torch.testing.assert_close(
            got.detach().cpu().double(), want, atol=3e-5, rtol=3e-4
        )
