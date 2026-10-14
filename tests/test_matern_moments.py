"""Independent native-site checks of sampled adjacent Gram preparation."""

from itertools import pairwise

import pytest
import torch

from torchcst._backends.torch.profiles.matern_moments import (
    evaluate_gram,
    static_tables,
)
from torchcst.kernels.profiles.matern import MaternMixtureSpec


def native_profile(delta, knot, period, size):
    """Actual sampled image sum, independent of orientation and moment tables."""
    d = delta + torch.arange(size, dtype=torch.float64) * (period / size)
    b = knot**2
    q = torch.exp(d.new_tensor(-period / b))
    denominator = -torch.expm1(d.new_tensor(-period / b))

    def half(r):
        return torch.exp(-r / b) * (
            (1 + r / b) / denominator + (period / b) * q / denominator.square()
        )

    peak = (1 + q) / denominator + 2 * (period / b) * q / denominator.square()
    return (half(d) + half(period - d)) / peak


def native_gram(spec, delta, period, size):
    profiles = [native_profile(delta, knot, period, size) for knot in spec.width_knots]
    return torch.stack(
        [
            torch.stack((a.square().sum(), (a * b).sum(), b.square().sum()))
            for a, b in pairwise(profiles)
        ]
    )


@pytest.mark.parametrize("count", [8, 16])
@pytest.mark.parametrize("period,size", [(102.4, 1024), (100.0, 1000), (1.0, 7)])
def test_gram_matches_independent_native_sites_and_derivative(count, period, size):
    spec = MaternMixtureSpec(
        width_knots=tuple(0.1 * 100 ** (j / (count - 1)) for j in range(count))
    )
    tables = static_tables(spec, size, period)
    h = period / size
    for fraction in [0.0, 0.001, 0.17, 0.499, 0.5, 0.83, 0.999, 1.0]:
        delta = torch.tensor(fraction * h, dtype=torch.float64, requires_grad=True)
        actual, derivative = evaluate_gram(spec, delta, period, size, static=tables)
        expected = native_gram(spec, delta, period, size)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=3e-13)
        native_derivative = torch.stack(
            [
                torch.autograd.grad(entry, delta, retain_graph=True)[0]
                for entry in expected.flatten()
            ]
        ).reshape_as(expected)
        torch.testing.assert_close(
            derivative, native_derivative, atol=2e-10, rtol=2e-10
        )
        computed_derivative = torch.stack(
            [
                torch.autograd.grad(entry, delta, retain_graph=True)[0]
                for entry in actual.flatten()
            ]
        ).reshape_as(expected)
        torch.testing.assert_close(
            derivative, computed_derivative, atol=3e-10, rtol=2e-10
        )


def test_batched_delta_and_grid_endpoints():
    spec = MaternMixtureSpec(width_knots=(0.1, 0.2, 1.0, 10.0))
    delta = torch.tensor([[0, 0.05], [0.07, 0.1]], dtype=torch.float64)
    grams, derivatives = evaluate_gram(spec, delta, 100.0, 1000)
    assert grams.shape == derivatives.shape == (2, 2, 3, 3)
    expected = torch.stack([native_gram(spec, d, 100.0, 1000) for d in delta.flatten()])
    torch.testing.assert_close(
        grams, expected.reshape_as(grams), atol=2e-11, rtol=3e-13
    )
    torch.testing.assert_close(grams[0, 0], grams[1, 1], atol=2e-11, rtol=3e-13)
    torch.testing.assert_close(
        derivatives[0, 0], derivatives[1, 1], atol=2e-10, rtol=2e-10
    )


@pytest.mark.parametrize("fraction", [0.0, 0.13, 0.5, 0.79, 1.0])
def test_convex_mixture_l2_includes_cross_term_and_center_derivative(fraction):
    spec = MaternMixtureSpec(width_knots=(0.1, 0.3, 1.0, 10.0))
    period, size = 102.4, 1024
    delta = torch.tensor(0.032, dtype=torch.float64, requires_grad=True)
    gram, derivative = evaluate_gram(spec, delta, period, size)
    weights = delta.new_tensor(
        ((1 - fraction) ** 2, 2 * fraction * (1 - fraction), fraction**2)
    )
    norm = (gram * weights).sum(dim=-1)
    norm_derivative = (derivative * weights).sum(dim=-1)
    profiles = [native_profile(delta, knot, period, size) for knot in spec.width_knots]
    expected = torch.stack(
        [
            ((1 - fraction) * left + fraction * right).square().sum()
            for left, right in pairwise(profiles)
        ]
    )
    torch.testing.assert_close(norm, expected, atol=2e-11, rtol=3e-13)
    expected_derivative = torch.stack(
        [torch.autograd.grad(entry, delta, retain_graph=True)[0] for entry in expected]
    )
    torch.testing.assert_close(
        norm_derivative, expected_derivative, atol=2e-10, rtol=2e-10
    )


def test_static_tables_use_fp64_and_have_expected_layout():
    spec = MaternMixtureSpec(width_knots=(0.1, 1.0, 10.0))
    constants, moments = static_tables(spec, 1000, 100.0)
    assert constants.shape == (3, 7)
    assert moments.shape == (5, 4, 3)
    assert constants.dtype == moments.dtype == torch.float64
    assert constants.device.type == moments.device.type == "cpu"
    assert not constants.requires_grad and not moments.requires_grad
    assert torch.isfinite(constants).all() and torch.isfinite(moments).all()
    assert (moments >= 0).all()
    assert constants[0, 5] == 0  # Underflowed image weight is valid for narrow width.


@pytest.mark.parametrize("size,period", [(0, 100.0), (1, 0), (1, float("inf"))])
def test_invalid_static_geometry_rejected(size, period):
    with pytest.raises(ValueError):
        static_tables(MaternMixtureSpec(), size, period)
