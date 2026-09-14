import torch

from torchcst.optim import (
    AtomGradientObservation,
    DenominatorMoment,
    ExpandedDenominatorMoment,
    ExpandedNumeratorMoment,
    NormalizedFixedPointSolver,
    NormalizedSolveResult,
    NormalizedSolver,
    NormalizedUpdateProblem,
    NumeratorMoment,
)


def make_expanded_moments(
    g: torch.Tensor,
    H: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[ExpandedNumeratorMoment, ExpandedDenominatorMoment]:
    observation = AtomGradientObservation(jg=g, gh=H, contributions=1)
    numerator = NumeratorMoment(0.0).expand(
        NumeratorMoment(0.0).initialize(g), observation, next_step=1
    )
    denominator_moment = DenominatorMoment(0.0, eps=eps)
    denominator = denominator_moment.expand(
        denominator_moment.initialize(g), observation, next_step=1
    )
    return numerator, denominator


def test_normalized_solver_reduces_to_one_rmsprop_step_when_h_is_zero() -> None:
    g = torch.tensor([[2.0, -0.5, 0.25]], dtype=torch.float64)
    H = torch.zeros(1, 3, 3, dtype=torch.float64)
    eps = 1e-6
    learning_rate = 0.1
    numerator, denominator = make_expanded_moments(g, H, eps=eps)
    problem = NormalizedUpdateProblem(
        numerator,
        denominator,
        learning_rate=learning_rate,
    )

    result = NormalizedFixedPointSolver(max_iter=8).solve(
        problem,
        trust_radius=10.0,
    )

    expected = -learning_rate * g / (g.abs() + eps)
    assert isinstance(result, NormalizedSolveResult)
    torch.testing.assert_close(result.displacement, expected)
    assert result.converged
    assert result.iterations == 2


def test_fixed_point_solver_implements_replaceable_solver_contract() -> None:
    assert isinstance(NormalizedFixedPointSolver(), NormalizedSolver)


def test_normalized_solver_projects_to_one_global_trust_ball() -> None:
    g = torch.ones(2, 3, dtype=torch.float64)
    H = torch.zeros(2, 3, 3, dtype=torch.float64)
    numerator, denominator = make_expanded_moments(g, H)
    problem = NormalizedUpdateProblem(numerator, denominator, learning_rate=1.0)
    radius = 0.25

    result = NormalizedFixedPointSolver(max_iter=8).solve(
        problem,
        trust_radius=radius,
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(result.displacement),
        torch.tensor(radius, dtype=result.displacement.dtype),
    )
    assert result.on_boundary
    assert result.converged
