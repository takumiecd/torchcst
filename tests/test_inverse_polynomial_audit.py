import torch

from experiments.analyze_inverse_polynomial import polynomial_residual


def test_neumann_matches_explicit_fixed_step_updates():
    values = torch.tensor([0.2, 1.0, 3.0], dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    x = torch.zeros_like(b)
    for degree in range(1, 9):
        x += 2 / (values[0] + values[-1]) * (b - values * x)
        expected = polynomial_residual(values, degree, "neumann") * b
        torch.testing.assert_close(b - values * x, expected)


def test_chebyshev_matches_direct_polynomial_and_scalar_inverse():
    values = torch.tensor([0.2, 0.7, 2.0], dtype=torch.float64)
    d, c = (values[-1] + values[0]) / 2, (values[-1] - values[0]) / 2
    t = (d - values) / c
    old, current = torch.ones_like(t), t
    denom_old, denom = torch.ones_like(d), d / c
    for degree in range(1, 9):
        torch.testing.assert_close(
            polynomial_residual(values, degree, "chebyshev"), current / denom
        )
        old, current = current, 2 * t * current - old
        denom_old, denom = denom, 2 * d / c * denom - denom_old
    torch.testing.assert_close(
        polynomial_residual(torch.ones(3), 4, "chebyshev"), torch.zeros(3)
    )
