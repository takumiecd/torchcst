"""Cell-mass math gates independent of the CUDA tiling and storage policy."""

import math

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


def axis(sites=128, half_width=24.0, spacing=1.0, origin=-0.25):
    return CellMassAxisSpec(
        profile=ChebyshevSquareSpec(modes=8, tail_bound=1e-3),
        site_count=sites,
        spacing=spacing,
        support_half_width=half_width,
        origin=origin,
    )


def quadrature_profile(spec, center, sigma):
    """Scalar native density plus Gauss integration, without Fourier coefficients."""
    m, period = spec.profile.modes, spec.profile_period
    tile = period / 3
    radius0 = (
        period
        / math.pi
        * math.acos(1 / math.cosh(math.acosh(spec.profile.tail_bound**-0.5) / m))
    )
    radius = (radius0**4 + (1 - (radius0 / tile) ** 4) * sigma**4) ** 0.25
    beta = 1 / math.cos(math.pi * radius / period)
    chebyshev = np.zeros(m + 1)
    chebyshev[m] = 1

    def density(t):
        arg = beta * np.cos(math.pi * t / period)
        h = np.polynomial.chebyshev.chebval(arg, chebyshev)
        h /= math.cosh(m * math.acosh(beta))
        return h * h

    points, weights = np.polynomial.legendre.leggauss(96)

    def integral(lo, hi):
        if hi <= lo:
            return 0.0
        return (
            (hi - lo)
            / 2
            * np.dot(weights, density((lo + hi) / 2 + (hi - lo) / 2 * points))
        )

    mass = integral(-spec.support_half_width, spec.support_half_width)
    center = (center - spec.origin) % spec.period
    images = math.ceil(spec.support_half_width / spec.period) + 1
    values = []
    for i in range(spec.site_count):
        area = 0.0
        for image in range(-images, images + 1):
            c = center + image * spec.period
            lo = max((i - 0.5) * spec.spacing - c, -spec.support_half_width)
            hi = min((i + 0.5) * spec.spacing - c, spec.support_half_width)
            area += integral(lo, hi)
        values.append(area / mass)
    return np.array(values)


@pytest.mark.parametrize("half_width", [0.15, 3.7, 64.0, 170.0])
def test_cell_mass_nonnegative_partition_and_true_periodization(half_width):
    spec = axis(half_width=half_width)
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


def test_128_forward_dx_and_all_physical_atom_gradients():
    spec = CellMassLinearSpec(input=axis(), output=axis(half_width=32.0))
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

    # All four coordinate directions, including both sigma paths, are checked
    # against the scalar quadrature oracle for ordinary and wraparound atoms.
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
    spec = CellMassLinearSpec(input=axis(8, 1.7), output=axis(8, 2.3, origin=0.2))
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
    "name,value", [("spacing", 0), ("origin", float("inf")), ("site_count", True)]
)
def test_invalid_axis(name, value):
    kwargs = {
        "profile": ChebyshevSquareSpec(modes=8),
        "site_count": 128,
        "spacing": 1,
        "support_half_width": 24,
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
