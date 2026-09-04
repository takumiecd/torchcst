import pytest
import torch
from torch import Tensor

from torchcst import Chart, CSTLinear, FullQuartic, Gaussian, Kernel, Separable
from torchcst.optim import (
    AcceptedFrameFirstMoment,
    AtomGradientObservation,
    ImplicitLinearAtomGrad,
    MomentContext,
    MomentSystem,
    QuarticProblem,
    SeparableDiagonalSecondMoment,
)


def make_nonlinear_site() -> CSTLinear:
    torch.manual_seed(31)
    return CSTLinear(
        Chart.linspace(4),
        Chart.linspace(3),
        atoms=2,
        kernel=Separable(
            input_profile=Gaussian(0.8),
            output_profile=Gaussian(0.6),
        ),
        dtype=torch.float64,
    )


def make_problem() -> QuarticProblem:
    site = make_nonlinear_site()
    system = MomentSystem(
        first=AcceptedFrameFirstMoment(0.9),
        second=SeparableDiagonalSecondMoment(0.99),
    )
    atom_grad = ImplicitLinearAtomGrad(
        mode="custom",
        request=system.observation_request,
    )
    site.atoms.set_grad(atom_grad)
    inputs = torch.randn(5, site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(5, site.out_features, dtype=torch.float64)
    atom_grad.begin()
    (site(inputs) * output_gradient).sum().backward()
    atom_grad.complete()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    expanded = system.expand(system.initialize(context), atom_grad.snapshot(), context)
    return QuarticProblem(context, expanded, learning_rate=0.2)


def test_quartic_value_matches_the_explicit_full_visible_objective() -> None:
    problem = make_problem()
    displacement = 0.03 * torch.randn_like(problem.context.current_point)

    represented = problem.context.geometry.displacement(
        displacement,
        point=problem.context.current_point,
    )
    first = (
        problem.moments.first.corrected.constant * displacement
    ).sum() + 0.5 * torch.einsum(
        "kp,kpq,kq->",
        displacement,
        problem.moments.first.corrected.linear,
        displacement,
    )
    diagonal = problem.moments.second.metric.diagonal()
    expected = first + (diagonal * represented.square()).sum() / (
        2.0 * problem.learning_rate
    )

    torch.testing.assert_close(problem.value(displacement), expected)


def test_quartic_gradient_matches_autograd_of_its_value() -> None:
    problem = make_problem()
    displacement = (
        0.02 * torch.randn_like(problem.context.current_point)
    ).requires_grad_(True)

    expected = torch.autograd.grad(problem.value(displacement), displacement)[0]
    actual = problem.gradient(displacement.detach())

    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)


class LinearAmplitudeKernel(Kernel):
    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return 1

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: str,
    ) -> Tensor:
        del output_chart, mode
        return input_chart.coordinates.new_zeros(atoms, 1)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        shape = (p.shape[0], output_chart.features, input_chart.features)
        return p[:, :1, None].expand(shape)


def make_convex_quadratic_problem() -> tuple[QuarticProblem, float]:
    site = CSTLinear(
        Chart.linspace(2),
        Chart.linspace(2),
        atoms=1,
        kernel=LinearAmplitudeKernel(),
        dtype=torch.float64,
    )
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    system = MomentSystem(
        first=AcceptedFrameFirstMoment(0.0),
        second=SeparableDiagonalSecondMoment(0.0, eps=1e-8),
    )
    observation = AtomGradientObservation(
        jg=torch.tensor([[-2.0]], dtype=torch.float64),
        gh=torch.zeros(1, 1, 1, dtype=torch.float64),
        row_square=torch.ones(2, dtype=torch.float64),
        column_square=torch.ones(2, dtype=torch.float64),
        contributions=1,
    )
    expanded = system.expand(system.initialize(context), observation, context)
    learning_rate = 0.2
    problem = QuarticProblem(
        context,
        expanded,
        learning_rate=learning_rate,
    )
    expected = 2.0 * learning_rate / (4.0 * (1.0 + 1e-8))
    return problem, expected


def test_full_quartic_solves_a_known_interior_problem() -> None:
    problem, expected = make_convex_quadratic_problem()
    point_before = problem.context.geometry.current_point()
    solver = FullQuartic(
        starts=3,
        max_iter=60,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    result = solver.solve(problem, trust_radius=0.5)

    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[expected]], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    assert result.objective < problem.value(torch.zeros_like(result.displacement))
    assert torch.linalg.vector_norm(result.displacement) <= 0.5
    assert result.projected_gradient_norm < 1e-7
    torch.testing.assert_close(
        problem.context.geometry.current_point(),
        point_before,
    )


def test_full_quartic_respects_the_trust_region_boundary() -> None:
    problem, _ = make_convex_quadratic_problem()
    solver = FullQuartic(starts=3, max_iter=60)

    result = solver.solve(problem, trust_radius=0.03)

    torch.testing.assert_close(
        result.displacement,
        torch.tensor([[0.03]], dtype=torch.float64),
        rtol=1e-4,
        atol=1e-8,
    )
    assert result.on_boundary
    assert result.projected_gradient_norm < 3e-6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"starts": 0}, "starts must be positive"),
        ({"max_iter": 0}, "max_iter must be positive"),
        ({"history_size": 0}, "history_size must be positive"),
        ({"tolerance_grad": 0.0}, "tolerances must be positive"),
    ],
)
def test_full_quartic_validates_its_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FullQuartic(**kwargs)


def test_full_quartic_requires_a_positive_trust_radius() -> None:
    problem, _ = make_convex_quadratic_problem()

    with pytest.raises(ValueError, match="trust_radius must be positive"):
        FullQuartic().solve(problem, trust_radius=0.0)
