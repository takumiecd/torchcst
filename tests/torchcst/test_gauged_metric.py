"""The pullback metric of L2-normalised kernel columns."""

from __future__ import annotations

import pytest
import torch

from torchcst.optim import metric
from torchcst.representation import (
    GaussianKernel,
    L2NormalizedColumns,
    TriangularKernel,
)


def _autograd_column_diagonal(kernel, mu, centers):
    gauge = L2NormalizedColumns()
    rows = []
    for center in centers:
        point = center.detach().clone().requires_grad_(True)

        def delivered(value):
            return gauge.columns(kernel, mu, value[None, :])[:, 0]

        jacobian = torch.autograd.functional.jacobian(delivered, point)
        rows.append(jacobian.square().sum(0))
    return torch.stack(rows)


@pytest.mark.parametrize("dimension", [1, 2])
def test_normalized_diagonal_matches_the_delivered_columns_autograd(dimension):
    kernel = GaussianKernel(0.3).double()
    generator = torch.Generator().manual_seed(31 + dimension)
    mu = torch.rand(9, dimension, generator=generator, dtype=torch.float64)
    centers = torch.rand(3, dimension, generator=generator, dtype=torch.float64)

    unit, factor = metric.normalized_columns(kernel, mu, centers)
    actual = metric.projected_directional(
        unit, factor, mu, centers, None, per_axis=True
    )
    expected = _autograd_column_diagonal(kernel, mu, centers)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)


def test_a_far_gaussian_keeps_a_finite_positive_normalized_metric():
    kernel = GaussianKernel(0.25)
    mu = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    centers = torch.tensor([[6.0]])

    raw, raw_factor = metric.columns(kernel, mu, centers)
    assert float(raw.abs().max()) == 0.0
    assert float(raw_factor.abs().max()) == 0.0

    unit, factor = metric.normalized_columns(kernel, mu, centers)
    diagonal = metric.projected_directional(
        unit, factor, mu, centers, None, per_axis=True
    )
    assert bool(torch.isfinite(diagonal).all())
    assert float(diagonal.min()) > 0.0


def test_a_compact_zero_column_has_zero_metric_not_nan():
    kernel = TriangularKernel(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).reshape(-1, 1)
    centers = torch.tensor([[4.0]], dtype=torch.float64)

    unit, factor = metric.normalized_columns(kernel, mu, centers)
    diagonal = metric.projected_directional(
        unit, factor, mu, centers, None, per_axis=True
    )
    assert bool(torch.isfinite(unit).all())
    assert bool(torch.isfinite(factor).all())
    assert bool(torch.isfinite(diagonal).all())
    assert bool((diagonal == 0).all())


def test_full_atom_diagonal_factorises_over_its_two_normalized_sides():
    kernel = GaussianKernel(0.4).double()
    mu_in = torch.tensor([[0.0], [0.3], [0.9]], dtype=torch.float64)
    mu_out = torch.tensor([[0.1], [0.8]], dtype=torch.float64)
    source = torch.tensor([[0.25]], dtype=torch.float64)
    target = torch.tensor([[0.65]], dtype=torch.float64)
    mass = torch.tensor([1.7], dtype=torch.float64)
    gauge = L2NormalizedColumns()

    unit_in, factor_in = metric.normalized_columns(kernel, mu_in, source)
    unit_out, factor_out = metric.normalized_columns(kernel, mu_out, target)
    actual_s, actual_t = metric.gauged_jacobian_sq(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass.square(),
        per_axis=True,
    )

    def represented(s, t):
        column_in = gauge.columns(kernel, mu_in, s[None, :])[:, 0]
        column_out = gauge.columns(kernel, mu_out, t[None, :])[:, 0]
        return mass[0] * column_out[:, None] * column_in[None, :]

    expected_s = torch.autograd.functional.jacobian(
        lambda s: represented(s, target[0]), source[0]
    ).square().sum((0, 1))
    expected_t = torch.autograd.functional.jacobian(
        lambda t: represented(source[0], t), target[0]
    ).square().sum((0, 1))
    torch.testing.assert_close(actual_s[0], expected_s, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(actual_t[0], expected_t, rtol=1e-9, atol=1e-12)


def test_traffic_weights_the_already_projected_normalized_derivative():
    kernel = GaussianKernel(0.4).double()
    mu_in = torch.tensor([[0.0], [0.3], [0.9]], dtype=torch.float64)
    mu_out = torch.tensor([[0.1], [0.8]], dtype=torch.float64)
    source = torch.tensor([[0.25]], dtype=torch.float64)
    target = torch.tensor([[0.65]], dtype=torch.float64)
    mass = torch.tensor([1.7], dtype=torch.float64)
    traffic_in = torch.tensor(
        [[1.0, -0.2, 0.3], [0.1, 0.8, -0.4]], dtype=torch.float64
    )
    traffic_out = torch.tensor(
        [[0.7, -0.1], [-0.3, 1.2]], dtype=torch.float64
    )
    gauge = L2NormalizedColumns()

    unit_in, factor_in = metric.normalized_columns(kernel, mu_in, source)
    unit_out, factor_out = metric.normalized_columns(kernel, mu_out, target)
    actual_s, actual_t = metric.gauged_jacobian_sq(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass.square(),
        traffic_in=traffic_in,
        traffic_out=traffic_out,
        per_axis=True,
    )

    jac_in = torch.autograd.functional.jacobian(
        lambda s: gauge.columns(kernel, mu_in, s[None, :])[:, 0], source[0]
    )
    jac_out = torch.autograd.functional.jacobian(
        lambda t: gauge.columns(kernel, mu_out, t[None, :])[:, 0], target[0]
    )
    in_column = gauge.columns(kernel, mu_in, source)[:, 0]
    out_column = gauge.columns(kernel, mu_out, target)[:, 0]
    expected_s = (
        mass.square()
        * (traffic_out @ out_column).square().sum() / traffic_out.shape[0]
        * (traffic_in @ jac_in).square().sum(0) / traffic_in.shape[0]
    )
    expected_t = (
        mass.square()
        * (traffic_in @ in_column).square().sum() / traffic_in.shape[0]
        * (traffic_out @ jac_out).square().sum(0) / traffic_out.shape[0]
    )
    torch.testing.assert_close(actual_s[0], expected_s)
    torch.testing.assert_close(actual_t[0], expected_t)
