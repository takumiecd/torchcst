from __future__ import annotations

import torch

from torchcst.optim import (
    AtomGradientObservation,
    ExpandedNumeratorMoment,
    NormalizedBoxFixedPointSolver,
    NormalizedFixedPointSolver,
    NormalizedUpdateProblem,
    NumeratorMoment,
    QuadraticBoxGradientSolver,
    QuadraticGradientSolver,
    QuadraticTrustSolver,
)
from torchcst.optim.moments import UnitDenominator
from torchcst.optim.solvers.taylor import _batched_spectral_ball_minimum


def make_numerator(g: torch.Tensor, H: torch.Tensor) -> ExpandedNumeratorMoment:
    observation = AtomGradientObservation(jg=g, gh=H, contributions=1)
    return NumeratorMoment(0.0).expand(
        NumeratorMoment(0.0).initialize(g), observation, next_step=1
    )


def make_unit_denominator(g: torch.Tensor, H: torch.Tensor):
    observation = AtomGradientObservation(jg=g, gh=H, contributions=1)
    denominator = UnitDenominator()
    return denominator.expand(denominator.initialize(g), observation, next_step=1)


def make_unit_problem(
    g: torch.Tensor,
    H: torch.Tensor,
    *,
    learning_rate: float = 0.1,
) -> NormalizedUpdateProblem:
    return NormalizedUpdateProblem(
        make_numerator(g, H),
        make_unit_denominator(g, H),
        learning_rate=learning_rate,
    )


def test_quadratic_solver_escapes_a_maximum_when_gradient_vanishes() -> None:
    g = torch.zeros(1, 2, dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[-2.0, 1.0]], dtype=torch.float64))
    problem = make_unit_problem(g, H)
    radius = 0.5

    result = QuadraticTrustSolver().solve(problem, trust_radius=radius)

    torch.testing.assert_close(
        torch.linalg.vector_norm(result.displacement[0]),
        torch.tensor(radius, dtype=torch.float64),
    )
    assert result.displacement[0, 0].abs() > result.displacement[0, 1].abs()
    assert result.solver_mode == "quadratic"
    assert result.on_boundary


def test_quadratic_solver_stays_at_a_positive_definite_critical_point() -> None:
    g = torch.zeros(1, 2, dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[2.0, 1.0]], dtype=torch.float64))
    problem = make_unit_problem(g, H)

    result = QuadraticTrustSolver().solve(problem, trust_radius=0.5)

    torch.testing.assert_close(result.displacement, torch.zeros_like(g))
    assert not result.on_boundary


def test_quadratic_solver_takes_the_interior_newton_step() -> None:
    g = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    problem = make_unit_problem(g, H)

    result = QuadraticTrustSolver().solve(problem, trust_radius=10.0)

    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
    )


def test_quadratic_solver_matches_the_single_system_spectral_minimum() -> None:
    g = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    H = torch.tensor([[[1.0, 0.2], [0.2, -0.5]]], dtype=torch.float64)
    problem = make_unit_problem(g, H, learning_rate=1.0)
    radius = 0.4

    result = QuadraticTrustSolver().solve(problem, trust_radius=radius)
    values, vectors = torch.linalg.eigh(0.5 * (H[0] + H[0].T))
    expected = _batched_spectral_ball_minimum(
        values.unsqueeze(0), vectors.unsqueeze(0), -g, radius
    )[0]

    torch.testing.assert_close(result.displacement[0], expected)


def test_gradient_solver_first_step_subtracts_the_normalized_gradient() -> None:
    g = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    eta = 0.1
    problem = make_unit_problem(g, H, learning_rate=eta)

    result = QuadraticGradientSolver(max_iter=1).solve(problem, trust_radius=10.0)

    torch.testing.assert_close(result.displacement, -eta * g)
    assert result.solver_mode == "gradient"
    assert result.iterations == 1


def test_gradient_solver_reaches_the_critical_point_of_m() -> None:
    g = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    eta = 0.1
    problem = make_unit_problem(g, H, learning_rate=eta)
    newton = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)

    result = QuadraticGradientSolver(max_iter=128, tolerance=1e-10).solve(
        problem,
        trust_radius=10.0,
    )

    torch.testing.assert_close(result.displacement, newton, atol=1e-6, rtol=0.0)
    assert result.converged
    assert not result.on_boundary


def test_gradient_solver_does_not_solve_the_implicit_normalized_map() -> None:
    g = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    H = torch.diag_embed(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    eta = 0.1
    problem = make_unit_problem(g, H, learning_rate=eta)

    gradient = QuadraticGradientSolver(max_iter=128, tolerance=1e-10).solve(
        problem,
        trust_radius=10.0,
    )
    implicit = NormalizedFixedPointSolver(max_iter=128, tolerance=1e-10).solve(
        problem,
        trust_radius=10.0,
    )

    torch.testing.assert_close(
        gradient.displacement,
        torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        implicit.displacement,
        -eta * g / (1.0 + eta * torch.tensor([[2.0, 4.0]], dtype=torch.float64)),
        atol=1e-6,
        rtol=0.0,
    )
    assert not torch.allclose(gradient.displacement, implicit.displacement, atol=1e-3)


def test_gradient_solver_walks_to_the_boundary_when_hessian_vanishes() -> None:
    g = torch.tensor([[3.0, 0.3]], dtype=torch.float64)
    H = torch.zeros(1, 2, 2, dtype=torch.float64)
    eta = 0.1
    radius = 0.5
    problem = make_unit_problem(g, H, learning_rate=eta)

    result = QuadraticGradientSolver(max_iter=64).solve(
        problem,
        trust_radius=radius,
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(result.displacement),
        torch.tensor(radius, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.displacement,
        -radius * g / torch.linalg.vector_norm(g),
    )
    assert result.on_boundary


def test_box_gradient_solver_uses_the_normalized_box_and_the_gradient_step() -> None:
    g = torch.ones(2, 3, dtype=torch.float64)
    H = torch.zeros(2, 3, 3, dtype=torch.float64)
    problem = make_unit_problem(g, H, learning_rate=0.1)
    bound = 0.25

    implicit = NormalizedBoxFixedPointSolver(max_iter=8).solve(
        problem,
        trust_radius=bound,
    )
    gradient = QuadraticBoxGradientSolver(max_iter=64).solve(
        problem,
        trust_radius=bound,
    )

    torch.testing.assert_close(implicit.displacement, -0.1 * g)
    torch.testing.assert_close(gradient.displacement, torch.full_like(g, -bound))
    assert gradient.solver_mode == "gradient"
    assert gradient.on_boundary
    assert not torch.allclose(gradient.displacement, implicit.displacement)
