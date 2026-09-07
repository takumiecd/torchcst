import pytest
import torch
from test_quartic import make_problem

from torchcst.optim.solvers._compiled import CompiledQuarticModel
from torchcst.optim.solvers._ray import ray_coefficients, unit_quartic_minimum


def reference(c):
    derivative = torch.tensor([4 * c[3], 3 * c[2], 2 * c[1], c[0]], dtype=torch.float64)
    while derivative.numel() > 1 and derivative[0].abs() < 1e-14:
        derivative = derivative[1:]
    degree = derivative.numel() - 1
    candidates = [0.0, 1.0]
    if degree:
        companion = torch.zeros(degree, degree, dtype=torch.float64)
        companion[0] = -derivative[1:] / derivative[0]
        if degree > 1:
            companion[1:, :-1] = torch.eye(degree - 1, dtype=torch.float64)
        for r in torch.linalg.eigvals(companion):
            if r.imag.abs() < 1e-7 and 0 <= r.real <= 1:
                candidates.append(float(r.real))
    return min(sum(float(c[k]) * t ** (k + 1) for k in range(4)) for t in candidates)


@pytest.mark.parametrize(
    "coefficients",
    [
        [0, 0, 0, 0],
        [-1, 1, 0, 0],
        [-1, 0, 0, 1],
        [1, -3, 2, 1],
        [0, 1, 0, 0],
        [-0.2, 0, 1, 0],
        [1, -1, 0, 0],
        [-0.1, 1, 1e-16, 1e-18],
    ],
)
def test_degenerate_and_interior_ray_minima(coefficients):
    c = torch.tensor(coefficients, dtype=torch.float64)
    t = unit_quartic_minimum(c)
    actual = sum(c[k] * t ** (k + 1) for k in range(4))
    assert 0 <= t <= 1
    torch.testing.assert_close(
        actual, torch.tensor(reference(c), dtype=c.dtype), atol=1e-9, rtol=1e-9
    )


def test_random_quartic_minima_against_companion_eigenvalues():
    torch.manual_seed(33)
    for _ in range(200):
        c = torch.randn(4, dtype=torch.float64)
        t = unit_quartic_minimum(c)
        actual = sum(c[k] * t ** (k + 1) for k in range(4))
        torch.testing.assert_close(
            actual, torch.tensor(reference(c), dtype=c.dtype), atol=1e-8, rtol=1e-8
        )


def test_ray_coefficients_reproduce_full_original_quartic():
    problem = make_problem()
    coefficients = CompiledQuarticModel(problem).coefficients
    x = torch.randn_like(coefficients[0]) * 0.03
    v = torch.randn_like(x) * 0.03
    c = ray_coefficients(*coefficients, x, v)
    for t in [0.0, 0.2, 0.5, 1.0]:
        expected = problem.value(x + t * v) - problem.value(x)
        actual = sum(c[k] * t ** (k + 1) for k in range(4))
        torch.testing.assert_close(actual, expected)


