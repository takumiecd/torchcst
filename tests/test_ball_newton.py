import pytest
import torch
from test_quadratic_feature_gram import make_pair
from test_quartic import make_convex_quadratic_problem, make_problem

from torchcst import BallNewton, SubspaceQuartic
from torchcst.optim.solvers.newton import _spectral_ball_minimum


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_quadratic_ball_handles_indefinite_hard_case_and_psd_nullspace(device):
    matrix = torch.diag(torch.tensor([-2.0, 1.0], dtype=torch.float64, device=device))
    rhs = torch.tensor([0.0, 0.3], dtype=torch.float64, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    result = _spectral_ball_minimum(eigenvalues, eigenvectors, rhs, 1.0)
    torch.testing.assert_close(
        result.abs(), torch.tensor([0.99**0.5, 0.1], dtype=torch.float64, device=device)
    )
    torch.testing.assert_close((matrix + 2 * torch.eye(2, device=device)) @ result, rhs)
    matrix = torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64, device=device))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    result = _spectral_ball_minimum(eigenvalues, eigenvectors, rhs, 1.0)
    torch.testing.assert_close(
        result, torch.tensor([0.0, 0.15], dtype=torch.float64, device=device)
    )


@pytest.mark.parametrize("radius", [0.03, 0.5])
def test_ball_newton_solves_known_interior_and_boundary(radius):
    problem, expected = make_convex_quadratic_problem()
    result = BallNewton(tolerance_grad=1e-8).solve(problem, trust_radius=radius)
    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[min(radius, expected)]], dtype=torch.float64),
    )
    assert result.converged
    assert result.projected_gradient_norm < 1e-8
    assert result.evaluations <= 100


def test_structured_hessian_matches_autograd_and_fallback():
    problem = make_problem()
    d = 0.15 * torch.randn_like(problem.context.current_point)
    expected = torch.func.hessian(problem.value)(d).reshape(d.numel(), d.numel())
    torch.testing.assert_close(problem.hessian(d), expected, rtol=1e-9, atol=1e-10)
    problem.context.geometry._cache_disabled = True
    torch.testing.assert_close(problem.hessian(d), expected, rtol=1e-9, atol=1e-10)


def test_ball_newton_decreases_original_quartic_without_mutating_point():
    problem = make_problem()
    point = problem.context.current_point.clone()
    result = BallNewton(max_iter=20).solve(problem, trust_radius=0.15)
    assert result.objective < 0
    assert torch.linalg.vector_norm(result.displacement) <= 0.15 + 1e-12
    torch.testing.assert_close(problem.value(result.displacement), result.objective)
    torch.testing.assert_close(problem.context.current_point, point)


def test_ball_newton_escapes_a_stationary_interior_maximum():
    class DoubleWell:
        def value_and_gradient(self, d):
            return (d**4 - d**2).sum(), 4 * d**3 - 2 * d

        def hessian(self, d):
            return torch.diag((12 * d**2 - 2).flatten())

    result = BallNewton(tolerance_grad=1e-8)._solve(
        DoubleWell(), torch.zeros(1, 1, dtype=torch.float64), 1.0
    )
    torch.testing.assert_close(
        result.displacement.abs(), torch.full((1, 1), 2**-0.5, dtype=torch.float64)
    )
    assert result.converged


def test_ball_newton_budget_and_validation():
    problem = make_problem()
    result = BallNewton(max_evaluations=1).solve(problem, trust_radius=0.1)
    assert result.evaluations == 1
    assert not result.converged
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BallNewton().solve(problem, trust_radius=invalid)
    with pytest.raises(ValueError):
        BallNewton(max_iter=0)
    with pytest.raises(ValueError):
        BallNewton(starts=0)


def test_multistart_newton_aggregates_work_and_keeps_best_objective():
    problem = make_problem()
    single = BallNewton(max_iter=12).solve(problem, trust_radius=0.15)
    multiple = BallNewton(starts=3, max_iter=12).solve(problem, trust_radius=0.15)
    assert multiple.objective <= single.objective + 1e-12
    assert multiple.evaluations >= single.evaluations
    assert 0 <= multiple.start_index < 3


def test_indefinite_nonhard_quadratic_satisfies_kkt():
    values = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64)
    vectors = torch.eye(3, dtype=torch.float64)
    rhs = torch.tensor([0.2, 0.3, -0.7], dtype=torch.float64)
    result = _spectral_ball_minimum(values, vectors, rhs, 1.0)
    multiplier = torch.dot(rhs - values * result, result)
    assert multiplier > 2.0
    torch.testing.assert_close(result.norm(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        (values + multiplier) * result, rhs, rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("dimension", [2, 6])
def test_restricted_quartic_keeps_value_gradient_and_hessian(dimension):
    problem = make_problem()
    basis, _ = torch.linalg.qr(
        torch.randn(
            problem.context.current_point.numel(), dimension, dtype=torch.float64
        )
    )
    model = problem.restricted_model(basis)
    y = 0.03 * torch.randn(1, dimension, dtype=torch.float64)
    d = (basis @ y.flatten()).reshape_as(problem.context.current_point)
    value, gradient = model.value_and_gradient(y)
    reference, reference_gradient = problem.value_and_gradient(d)
    torch.testing.assert_close(value, reference, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        gradient.flatten(),
        basis.T @ reference_gradient.flatten(),
        rtol=1e-9,
        atol=1e-10,
    )
    torch.testing.assert_close(
        model.hessian(y), basis.T @ problem.hessian(d) @ basis, rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("radius", [0.03, 0.5])
def test_subspace_solver_known_quadratic(radius):
    problem, expected = make_convex_quadratic_problem()
    result = SubspaceQuartic().solve(problem, trust_radius=radius)
    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[min(radius, expected)]], dtype=torch.float64),
    )
    assert result.converged


def test_subspace_solver_preserves_feasibility_and_original_objective():
    problem = make_problem()
    result = SubspaceQuartic(max_models=4, max_dimension=4).solve(
        problem, trust_radius=0.15
    )
    assert result.objective < 0
    assert result.displacement.norm() <= 0.15 + 1e-12
    torch.testing.assert_close(result.objective, problem.value(result.displacement))
    assert result.evaluations <= 1 + 4 * 101


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_hessian_with_amplitude_bandwidth(dtype, device):
    problem, _ = make_pair(dtype, True, device=device)
    d = 0.03 * torch.randn_like(problem.context.current_point)
    expected = torch.func.hessian(problem.value)(d).reshape(d.numel(), d.numel())
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        problem.hessian(d), expected, rtol=tolerance, atol=tolerance
    )
