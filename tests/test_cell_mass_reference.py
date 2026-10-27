"""Independent native-density integration and tile-window approximation gates."""

import math
from functools import lru_cache

import numpy as np
import pytest
import torch

from torchcst._backends.torch.kernels.cell_mass import (
    physical_linear_reference,
    profile_cell_masses,
)
from torchcst.kernels.cell_mass import CellMassAxisSpec, CellMassLinearSpec
from torchcst.kernels.normalization import NormalizationSpec
from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec


def axis(sites=128, tile_sites=16, spacing=1.0, origin=-0.25, period_tiles=3):
    return CellMassAxisSpec(
        profile=ChebyshevSquareSpec(modes=8, tail_bound=1e-3),
        site_count=sites,
        spacing=spacing,
        tile_sites=tile_sites,
        period_tiles=period_tiles,
        origin=origin,
    )


@lru_cache(maxsize=1)
def gauss_rule():
    return np.polynomial.legendre.leggauss(96)


def native_integrator(spec, sigma):
    """Scalar Chebyshev density and Gauss integration; no Fourier coefficients."""
    m, period = spec.profile.modes, spec.profile_period
    scale = period / 3
    radius0 = (
        period
        / math.pi
        * math.acos(1 / math.cosh(math.acosh(spec.profile.tail_bound**-0.5) / m))
    )
    radius = (radius0**4 + (1 - (radius0 / scale) ** 4) * sigma**4) ** 0.25
    beta = 1 / math.cos(math.pi * radius / period)
    chebyshev = np.zeros(m + 1)
    chebyshev[m] = 1

    def density(t):
        h = np.polynomial.chebyshev.chebval(
            beta * np.cos(math.pi * t / period), chebyshev
        )
        return (h / math.cosh(m * math.acosh(beta))) ** 2

    points, weights = gauss_rule()

    def integral(lo, hi):
        if hi <= lo:
            return 0.0
        return (
            (hi - lo)
            / 2
            * np.dot(weights, density((lo + hi) / 2 + (hi - lo) / 2 * points))
        )

    return density, integral


def quadrature_profile(spec, center, sigma):
    """Whole-cell oracle over one unwrapped period; scatter onto chart sites."""
    _, integral = native_integrator(spec, sigma)
    mass = integral(-spec.profile_period / 2, spec.profile_period / 2)
    center = (center - spec.origin) % spec.period
    owner = math.floor((center + spec.spacing / 2) / spec.tile_width)
    first = (owner - spec.period_tiles // 2) * spec.tile_sites
    result = np.zeros(spec.site_count)
    for site in range(first, first + spec.period_tiles * spec.tile_sites):
        lo = (site - 0.5) * spec.spacing - center
        result[site % spec.site_count] += integral(lo, lo + spec.spacing) / mass
    return result


def compact_quadrature_profile(spec, center, sigma):
    """Intended compact core only, for reporting the approximation error.

    Clipping is confined to this independent analysis oracle, never CUDA/Z/C.
    The compact core is separately normalized, so its difference includes both
    discarded tail mass and the normalization change.
    """
    _, integral = native_integrator(spec, sigma)
    radius = spec.support_half_width
    mass = integral(-radius, radius)
    full_mass = integral(-spec.profile_period / 2, spec.profile_period / 2)
    center = (center - spec.origin) % spec.period
    result = np.zeros(spec.site_count)
    for site in range(spec.site_count):
        for image in (-1, 0, 1):
            lo = max(
                (site - 0.5) * spec.spacing - center + image * spec.period, -radius
            )
            hi = min((site + 0.5) * spec.spacing - center + image * spec.period, radius)
            result[site] += integral(lo, hi) / mass
    return result, max(0.0, 1 - mass / full_mass), full_mass


@pytest.mark.parametrize(
    "tile_sites,period_tiles",
    [(1, 3), (2, 3), (8, 3), (16, 3), (16, 5), (32, 3), (128, 1)],
)
def test_whole_tile_partition_nonnegativity_and_chart_wrap(tile_sites, period_tiles):
    spec = axis(tile_sites=tile_sites, period_tiles=period_tiles)
    centers = torch.tensor([-256.37, -0.25, 0.13, 127.4, 512.7], dtype=torch.float64)
    sigma = centers.new_full(centers.shape, spec.profile_period / 12)
    actual = profile_cell_masses(spec, centers, sigma)
    assert actual.min() >= -2e-13
    torch.testing.assert_close(
        actual.sum(-1), torch.ones_like(sigma), atol=2e-12, rtol=0
    )
    torch.testing.assert_close(
        actual,
        profile_cell_masses(spec, centers + 3 * spec.period, sigma),
        atol=2e-12,
        rtol=0,
    )
    for row, center, width in zip(actual, centers, sigma):
        np.testing.assert_allclose(
            row.detach().numpy(),
            quadrature_profile(spec, float(center), float(width)),
            atol=3e-12,
            rtol=3e-11,
        )


def test_cell_edge_routing_and_exactly_zero_unselected_tiles():
    spec = axis()
    # A centre crosses a TILE CELL EDGE at origin + q*T - spacing/2.
    seam = spec.origin + spec.tile_width - spec.spacing / 2
    centers = torch.tensor([seam - 1e-5, seam + 1e-5], dtype=torch.float64)
    values = profile_cell_masses(spec, centers, centers.new_full((2,), 0.5))
    selected = [set(range(112, 128)) | set(range(32)), set(range(48))]
    for row, keep in zip(values, selected):
        outside = [i for i in range(128) if i not in keep]
        assert row[outside].count_nonzero() == 0
    # All retained cells sum to one through a full-period integral, not a
    # pointwise norm or a data-dependent normalization pass.
    torch.testing.assert_close(
        values.sum(-1), torch.ones(2, dtype=values.dtype), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("sigma_fraction", [0.001, 0.05, 0.2, 1 / 3])
def test_fourier_tail_approximation_against_compact_core(sigma_fraction):
    spec = axis()
    sigma = spec.profile_period * sigma_fraction
    for center in (-0.72, 8.1, 15.24, 127.9):
        approximate = quadrature_profile(spec, center, sigma)
        compact, tail_fraction, full_mass = compact_quadrature_profile(
            spec, center, sigma
        )
        l1_error = np.abs(approximate - compact).sum()
        assert l1_error <= 2 * tail_fraction + 3e-12
        # Native side lobes are bounded relative to density(0)=1. This is a
        # mass-error bound, not an assertion that Fourier tails are exactly zero.
        bound = (
            (spec.profile_period - 2 * spec.support_half_width)
            * spec.profile.tail_bound
            / full_mass
        )
        assert tail_fraction <= bound + 3e-12


def test_128_forward_dx_and_all_physical_atom_gradients():
    spec = CellMassLinearSpec(input=axis(), output=axis(tile_sites=32))
    rng = torch.Generator().manual_seed(120)
    atoms = torch.randn(128, 4, generator=rng, dtype=torch.float64)
    atoms[:, 1] = torch.linspace(0.05, 12, 128)
    atoms[:, 2:] *= 150
    atoms[0, 0] = 0
    atoms.requires_grad_()
    x = torch.randn(4, 128, generator=rng, dtype=torch.float64, requires_grad=True)
    dy = torch.randn(4, 128, generator=rng, dtype=torch.float64)
    y = physical_linear_reference(spec, x, atoms)
    dx, dp = torch.autograd.grad(y, (x, atoms), dy)
    inp = np.stack(
        [
            quadrature_profile(spec.input, float(a[3]), float(a[1]))
            for a in atoms.detach()
        ]
    )
    out = np.stack(
        [
            quadrature_profile(spec.output, float(a[2]), float(a[1]))
            for a in atoms.detach()
        ]
    )
    weight = (out.T * atoms[:, 0].detach().numpy()) @ inp
    np.testing.assert_allclose(
        y.detach().numpy(), x.detach().numpy() @ weight.T, atol=2e-11
    )
    np.testing.assert_allclose(dx.numpy(), dy.numpy() @ weight, atol=2e-11)
    assert torch.isfinite(dp).all()
    assert dp[0, 0].abs() > 1e-10
    torch.testing.assert_close(
        dp[0, 1:], torch.zeros(3, dtype=dp.dtype), rtol=0, atol=0
    )
    for a in (0, 3, 64, 127):
        for column in range(4):
            row = atoms[a].detach().numpy().copy()
            step = 2e-5

            def objective(sign, row=row, column=column, step=step):
                p = row.copy()
                p[column] += sign * step
                vi = quadrature_profile(spec.input, p[3], p[1])
                uo = quadrature_profile(spec.output, p[2], p[1])
                return p[0] * np.dot(x.detach().numpy() @ vi, dy.numpy() @ uo)

            expected = (objective(1) - objective(-1)) / (2 * step)
            assert float(dp[a, column]) == pytest.approx(expected, abs=2e-8, rel=2e-6)


def test_gradcheck_and_empty_atom_set():
    spec = CellMassLinearSpec(input=axis(8, 2), output=axis(8, 2, origin=0.2))
    atoms = torch.tensor(
        [[0.4, 0.3, 7.91, -0.12], [-0.3, 0.5, 2.27, 6.12]],
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, p: physical_linear_reference(spec, x, p), (x, atoms)
    )
    empty = torch.empty(0, 4, dtype=torch.float64, requires_grad=True)
    y = physical_linear_reference(spec, x, empty)
    dx, dp = torch.autograd.grad(y.sum(), (x, empty))
    assert y.count_nonzero() == 0 and dx.count_nonzero() == 0
    assert dp.shape == (0, 4)


@pytest.mark.parametrize(
    "name,value",
    [
        ("spacing", 0),
        ("origin", float("inf")),
        ("site_count", True),
        ("tile_sites", 3),
        ("period_tiles", 4),
        ("period_tiles", 9),
        ("period_tiles", 1),
    ],
)
def test_invalid_axis(name, value):
    kwargs = {
        "profile": ChebyshevSquareSpec(modes=8),
        "site_count": 128,
        "spacing": 1,
        "tile_sites": 16,
    }
    kwargs[name] = value
    with pytest.raises((TypeError, ValueError)):
        CellMassAxisSpec(**kwargs)


def test_reject_l2_and_invalid_physical_rows():
    with pytest.raises(ValueError, match="continuous"):
        CellMassLinearSpec(
            input=axis(), output=axis(), normalization=NormalizationSpec()
        )
    spec = CellMassLinearSpec(input=axis(), output=axis())
    x = torch.zeros(4, 128, dtype=torch.float64)
    p = torch.zeros(1, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="sigma"):
        physical_linear_reference(spec, x, p)
