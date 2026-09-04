import torch

from torchcst import Chart, CSTLinear, Gaussian, Separable
from torchcst.optim import (
    AcceptedFrameFirstMoment,
    AcceptedFrameFirstMomentState,
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
    MomentContext,
    MomentSystem,
    SeparableDiagonalSecondMoment,
    SeparableSecondMomentState,
)


def make_site() -> CSTLinear:
    torch.manual_seed(29)
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


def make_system() -> MomentSystem:
    return MomentSystem(
        first=AcceptedFrameFirstMoment(0.9, rtol=1e-12),
        second=SeparableDiagonalSecondMoment(0.99, eps=1e-8),
    )


def observe(site: CSTLinear, system: MomentSystem) -> AtomGradientObservation:
    atom_grad = ImplicitLinearAtomGrad(
        mode="custom",
        request=system.observation_request,
        row_chunk_size=2,
    )
    site.atoms.set_grad(atom_grad)
    inputs = torch.randn(5, site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(5, site.out_features, dtype=torch.float64)
    atom_grad.begin()
    (site(inputs) * output_gradient).sum().backward()
    atom_grad.complete()
    return atom_grad.snapshot()


def observation_from_visible_gradient(
    site: CSTLinear,
    point: torch.Tensor,
    visible_gradient: torch.Tensor,
) -> AtomGradientObservation:
    derivatives = site.cst_derivatives()
    return AtomGradientObservation(
        jg=derivatives.pullback(visible_gradient, parameter_point=point),
        gh=derivatives.contracted_hessian(
            visible_gradient,
            parameter_point=point,
        ),
        row_square=visible_gradient.square().mean(dim=1),
        column_square=visible_gradient.square().mean(dim=0),
        contributions=1,
    )


def test_moment_system_unions_its_autograd_requirements() -> None:
    system = make_system()

    assert system.observation_request == AtomGradRequest(
        jg=True,
        gh=True,
        row_square=True,
        column_square=True,
    )


def test_selective_atom_grad_only_computes_requested_observations() -> None:
    site = make_site()
    atom_grad = ImplicitLinearAtomGrad(
        request=AtomGradRequest(jg=True),
    )
    site.atoms.set_grad(atom_grad)
    atom_grad.begin()
    site(torch.randn(3, site.in_features, dtype=torch.float64)).sum().backward()
    atom_grad.complete()

    observation = atom_grad.snapshot()
    assert observation.jg is not None
    assert observation.gh is None
    assert observation.row_square is None
    assert observation.column_square is None


def test_first_expansion_and_second_metric_are_bias_corrected() -> None:
    site = make_site()
    system = make_system()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    state = system.initialize(context)
    observation = observe(site, system)

    expanded = system.expand(state, observation, context)

    assert observation.jg is not None
    assert observation.gh is not None
    assert observation.row_square is not None
    assert observation.column_square is not None
    torch.testing.assert_close(expanded.first.raw.constant, 0.1 * observation.jg)
    torch.testing.assert_close(expanded.first.raw.linear, 0.1 * observation.gh)
    torch.testing.assert_close(expanded.first.corrected.constant, observation.jg)
    torch.testing.assert_close(expanded.first.corrected.linear, observation.gh)
    torch.testing.assert_close(
        expanded.second.metric.row,
        observation.row_square,
    )
    torch.testing.assert_close(
        expanded.second.metric.column,
        observation.column_square,
    )


def test_compress_commits_both_moments_at_the_accepted_frame() -> None:
    site = make_site()
    system = make_system()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    state = system.initialize(context)
    observation = observe(site, system)
    expanded = system.expand(state, observation, context)
    accepted_displacement = 0.02 * torch.randn_like(context.current_point)

    next_state = system.compress(expanded, accepted_displacement, context)

    assert state.step == 0
    assert next_state.step == 1
    assert isinstance(next_state.first, AcceptedFrameFirstMomentState)
    assert isinstance(next_state.second, SeparableSecondMomentState)
    torch.testing.assert_close(
        next_state.first.frame.displacement,
        accepted_displacement,
    )
    numerator = expanded.first.raw.at(accepted_displacement)
    torch.testing.assert_close(
        geometry.gram(next_state.first.frame).matvec(next_state.first.alpha),
        numerator,
        rtol=1e-8,
        atol=1e-9,
    )
    assert observation.row_square is not None
    assert observation.column_square is not None
    torch.testing.assert_close(next_state.second.row, 0.01 * observation.row_square)
    torch.testing.assert_close(
        next_state.second.column,
        0.01 * observation.column_square,
    )


def test_next_expansion_transports_the_old_visible_first_moment() -> None:
    site = make_site()
    system = make_system()
    geometry = site.cst_frame_geometry()
    old_point = geometry.current_point()
    old_context = MomentContext(geometry, old_point)
    first_observation = observe(site, system)
    initial = system.initialize(old_context)
    first_expanded = system.expand(initial, first_observation, old_context)
    accepted_displacement = 0.015 * torch.randn_like(old_point)
    state = system.compress(
        first_expanded,
        accepted_displacement,
        old_context,
    )
    assert isinstance(state.first, AcceptedFrameFirstMomentState)

    current_point = old_point + accepted_displacement
    current_context = MomentContext(geometry, current_point)
    visible_gradient = torch.randn(
        site.out_features,
        site.in_features,
        dtype=old_point.dtype,
    )
    second_observation = observation_from_visible_gradient(
        site,
        current_point,
        visible_gradient,
    )
    expanded = system.expand(state, second_observation, current_context)
    trial_displacement = 0.01 * torch.randn_like(current_point)

    previous_visible = geometry.visible_pushforward(
        state.first.frame,
        state.first.alpha,
    )
    conceptual_raw_moment = 0.9 * previous_visible + 0.1 * visible_gradient
    expected = site.cst_derivatives().pullback(
        conceptual_raw_moment,
        at=trial_displacement,
        parameter_point=current_point,
    )
    torch.testing.assert_close(expanded.first.raw.at(trial_displacement), expected)


def test_expansion_is_provisional_and_does_not_mutate_persistent_state() -> None:
    site = make_site()
    system = make_system()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    state = system.initialize(context)
    observation = observe(site, system)
    first_alpha = state.first.alpha.clone()
    second_row = state.second.row.clone()
    second_column = state.second.column.clone()

    expanded = system.expand(state, observation, context)
    rejected_state = system.reject(expanded, state)

    assert rejected_state is state
    assert state.step == 0
    torch.testing.assert_close(state.first.alpha, first_alpha)
    torch.testing.assert_close(state.second.row, second_row)
    torch.testing.assert_close(state.second.column, second_column)


def test_separable_metric_matches_the_documented_dense_diagonal() -> None:
    site = make_site()
    system = make_system()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    observation = observe(site, system)
    expanded = system.expand(system.initialize(context), observation, context)
    metric = expanded.second.metric
    value = torch.randn(*metric.visible_shape, dtype=torch.float64)

    assert observation.row_square is not None
    assert observation.column_square is not None
    variance = (
        observation.row_square[:, None]
        * observation.column_square[None, :]
        / observation.row_square.mean()
    )
    expected_diagonal = variance.sqrt() + 1e-8

    torch.testing.assert_close(metric.diagonal(), expected_diagonal)
    torch.testing.assert_close(metric.apply(value), expected_diagonal * value)
    torch.testing.assert_close(metric.inner(value, value), (expected_diagonal * value.square()).sum())
