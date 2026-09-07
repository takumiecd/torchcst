import pytest
import torch
from test_quartic import make_problem

from torchcst.optim.solvers._compiled import (
    CompiledQuarticModel,
    spectral_ball_device,
    visible_hessian,
    visible_value_gradient,
)
from torchcst.optim.solvers.newton import _spectral_ball_minimum


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_quartic_derivatives_match_original(dtype):
    problem = make_problem()
    model = CompiledQuarticModel(problem, backend="eager", mode="default")
    coefficients = tuple(c.to(dtype) for c in model.coefficients)
    d = torch.randn_like(coefficients[0]) * 0.03
    value, gradient = visible_value_gradient(*coefficients, d)
    expected_gradient = torch.func.grad(
        lambda x: visible_value_gradient(*coefficients, x)[0]
    )(d)
    expected_hessian = torch.func.hessian(
        lambda x: visible_value_gradient(*coefficients, x)[0]
    )(d).reshape(d.numel(), d.numel())
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(visible_hessian(*coefficients, d), expected_hessian)
    if dtype == torch.float64:
        reference, reference_gradient = problem.value_and_gradient(d)
        torch.testing.assert_close(value, reference)
        torch.testing.assert_close(gradient, reference_gradient)
        torch.testing.assert_close(
            visible_hessian(*coefficients, d), problem.hessian(d)
        )
        torch.testing.assert_close(
            model.value_and_gradient(d), (reference, reference_gradient)
        )
        torch.testing.assert_close(model.hessian(d), problem.hessian(d))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "eigenvalues,rhs,radius",
    [
        ([-2.0, 1.0], [0.0, 0.3], 1.0),
        ([0.0, 2.0], [0.0, 0.3], 1.0),
        ([-2.0, 1.0, 3.0], [0.2, 0.3, -0.7], 1.0),
        ([1.0, 2.0], [0.2, -0.1], 0.01),
        ([1.0, 2.0], [0.2, -0.1], 1.0),
        ([0.0, 0.0], [0.0, 0.0], 1.0),
    ],
)
def test_device_spectral_matches_reference(dtype, eigenvalues, rhs, radius):
    values = torch.tensor(eigenvalues, dtype=dtype)
    vectors = torch.eye(len(eigenvalues), dtype=dtype)
    right = torch.tensor(rhs, dtype=dtype)
    expected = _spectral_ball_minimum(values, vectors, right, radius)
    actual = spectral_ball_device(values, vectors, right, radius)
    torch.testing.assert_close(actual, expected)
    assert actual.norm() <= radius * (1 + 10 * torch.finfo(dtype).eps)


def test_compiled_coefficients_change_without_stale_capture():
    problem = make_problem()
    model = CompiledQuarticModel(problem, backend="eager", mode="default")
    d = torch.full_like(problem.context.current_point, 0.01)
    before = model.value_and_gradient(d)
    changed = list(model.coefficients)
    changed[0] = changed[0] + 2
    model.coefficients = tuple(changed)
    after = model.value_and_gradient(d)
    torch.testing.assert_close(after[0] - before[0], 2 * d.sum())
    torch.testing.assert_close(after[1] - before[1], torch.full_like(d, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_compiled_kernels_and_replay_lifetime(dtype):
    from torchcst.optim.solvers._compiled import call_compiled

    model = CompiledQuarticModel(make_problem())
    coefficients = tuple(c.to(device="cuda", dtype=dtype) for c in model.coefficients)
    d = torch.full_like(coefficients[0], 0.01)
    first = call_compiled("visible_value_gradient", *coefficients, d)
    saved = tuple(x.clone() for x in first)
    for scale in (2.0, 3.0, 4.0):
        changed = d * scale
        actual = call_compiled("visible_value_gradient", *coefficients, changed)
        torch.testing.assert_close(
            actual, visible_value_gradient(*coefficients, changed)
        )
        torch.testing.assert_close(
            call_compiled("visible_hessian", *coefficients, changed),
            visible_hessian(*coefficients, changed),
        )
    torch.testing.assert_close(first, saved)
    values = torch.tensor([-2.0, 1.0, 3.0], device="cuda", dtype=dtype)
    vectors = torch.eye(3, device="cuda", dtype=dtype)
    rhs = torch.tensor([0.2, 0.3, -0.7], device="cuda", dtype=dtype)
    torch.testing.assert_close(
        call_compiled("spectral_ball_device", values, vectors, rhs, 1.0),
        _spectral_ball_minimum(values, vectors, rhs, 1.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_device_newton_budget_and_reference_diagnostics():
    from test_quadratic_feature_gram import make_pair

    from torchcst import BallNewton

    problem, _ = make_pair(
        torch.float64, True, input_size=5, output_size=4, device="cuda"
    )
    result = BallNewton(
        execution="compiled", secular_solver="device", max_iter=5, max_evaluations=20
    ).solve(problem, trust_radius=0.1)
    value, gradient = problem.value_and_gradient(result.displacement)
    torch.testing.assert_close(result.objective, value)
    torch.testing.assert_close(
        result.projected_gradient_norm,
        BallNewton._projected_norm(result.displacement, gradient, 0.1),
    )
    assert torch.isfinite(result.displacement).all()
    assert result.displacement.norm() <= 0.1 * (1 + 1e-10)
    assert result.objective <= problem.value(torch.zeros_like(result.displacement))
    assert result.evaluations <= 20
