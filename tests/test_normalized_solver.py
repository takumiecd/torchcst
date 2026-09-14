import torch

from torchcst.optim import (
    AtomGradientObservation,
    DenominatorMoment,
    ExpandedDenominatorMoment,
    ExpandedNumeratorMoment,
    NormalizedBoxFixedPointSolver,
    NormalizedFixedPointSolver,
    NormalizedSolver,
    NormalizedSolveResult,
    NormalizedUpdateProblem,
    NumeratorMoment,
)


class CountingEvaluation:
    def __init__(self, zero_value: float, displaced_value: float) -> None:
        self.zero = torch.tensor([[zero_value]], dtype=torch.float64)
        self.displaced = torch.tensor([[displaced_value]], dtype=torch.float64)
        self.zero_calls = 0
        self.displaced_calls = 0

    @property
    def point_shape(self) -> tuple[int, int]:
        return (1, 1)

    @property
    def device(self) -> torch.device:
        return self.zero.device

    @property
    def dtype(self) -> torch.dtype:
        return self.zero.dtype

    def at(self, displacement: torch.Tensor) -> torch.Tensor:
        del displacement
        self.displaced_calls += 1
        return self.displaced

    def at_zero(self) -> torch.Tensor:
        self.zero_calls += 1
        return self.zero


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


def test_first_solver_iteration_uses_zero_point_evaluation() -> None:
    numerator = CountingEvaluation(1.0, 3.0)
    denominator = CountingEvaluation(1.0, 1.0)
    problem = NormalizedUpdateProblem(
        numerator,
        denominator,
        learning_rate=1.0,
    )

    result = NormalizedFixedPointSolver(max_iter=2).solve(
        problem,
        trust_radius=10.0,
    )

    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[-3.0]], dtype=torch.float64),
    )
    assert numerator.zero_calls == 1
    assert denominator.zero_calls == 1
    assert numerator.displaced_calls == 2
    assert denominator.displaced_calls == 2


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


def test_normalized_box_solver_bounds_each_coordinate_independently() -> None:
    g = torch.ones(2, 3, dtype=torch.float64)
    H = torch.zeros(2, 3, 3, dtype=torch.float64)
    numerator, denominator = make_expanded_moments(g, H)
    problem = NormalizedUpdateProblem(numerator, denominator, learning_rate=1.0)
    bound = 0.25

    solver = NormalizedBoxFixedPointSolver(max_iter=8)
    result = solver.solve(problem, trust_radius=bound)

    torch.testing.assert_close(
        result.displacement,
        torch.full_like(g, -bound),
    )
    assert torch.linalg.vector_norm(result.displacement) > bound
    assert solver.displacement_is_valid(result.displacement, trust_radius=bound)
    assert not solver.displacement_is_valid(
        result.displacement - 1e-3,
        trust_radius=bound,
    )
    assert result.on_boundary
    assert result.converged
