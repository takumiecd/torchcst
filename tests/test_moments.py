import torch

from torchcst import Amplitude, Chart, CSTLinear, Gaussian, Separable
from torchcst.optim import (
    AcceptedFrameFirstMoment,
    AcceptedFrameFirstMomentState,
    AtomGradientObservation,
    AtomGradRequest,
    DenominatorMoment,
    DenominatorMomentState,
    DenominatorPolynomial,
    ExpandedDenominatorMoment,
    ExpandedNumeratorMoment,
    ImplicitLinearAtomGrad,
    MomentContext,
    MomentSystem,
    NumeratorMoment,
    NumeratorMomentState,
    SeparableDiagonalSecondMoment,
    SeparableSecondMomentState,
)


def make_site() -> CSTLinear:
    torch.manual_seed(29)
    return CSTLinear(
        Chart.linspace(4),
        Chart.linspace(3),
        atoms=2,
        kernel=Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.6),
            )
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


def test_numerator_moment_matches_affine_ema_and_bias_correction() -> None:
    torch.manual_seed(31)
    beta = 0.8
    moment = NumeratorMoment(beta)
    g = torch.randn(2, 3, dtype=torch.float64)
    H = torch.randn(2, 3, 3, dtype=torch.float64)
    observation = AtomGradientObservation(jg=g, gh=H, contributions=1)
    state = moment.initialize(g)

    expanded = moment.expand(state, observation, next_step=1)

    assert isinstance(expanded, ExpandedNumeratorMoment)
    torch.testing.assert_close(expanded.raw.constant, (1.0 - beta) * g)
    torch.testing.assert_close(expanded.raw.linear, (1.0 - beta) * H)
    torch.testing.assert_close(expanded.corrected.constant, g)
    torch.testing.assert_close(expanded.corrected.linear, H)

    displacement = torch.randn_like(g)
    torch.testing.assert_close(
        expanded.at(displacement),
        g + torch.einsum("kpq,kq->kp", H, displacement),
    )


def test_denominator_moment_matches_componentwise_squared_affine_gradient() -> None:
    torch.manual_seed(41)
    beta = 0.8
    eps = 1e-6
    moment = DenominatorMoment(beta, eps=eps)
    g = torch.randn(2, 3, dtype=torch.float64)
    H = torch.randn(2, 3, 3, dtype=torch.float64)
    observation = AtomGradientObservation(jg=g, gh=H, contributions=1)
    state = moment.initialize(g)

    expanded = moment.expand(state, observation, next_step=1)

    assert isinstance(expanded, ExpandedDenominatorMoment)
    assert isinstance(expanded.pending_state, DenominatorMomentState)
    assert isinstance(expanded.corrected, DenominatorPolynomial)
    current_Z = torch.einsum("kip,kiq->kipq", H, H)
    torch.testing.assert_close(expanded.raw.x, (1.0 - beta) * g.square())
    torch.testing.assert_close(
        expanded.raw.y,
        (1.0 - beta) * torch.einsum("ki,kip->kip", g, H),
    )
    torch.testing.assert_close(expanded.raw.Z, (1.0 - beta) * current_Z)
    torch.testing.assert_close(expanded.corrected.x, g.square())
    probe = torch.randn_like(g)
    torch.testing.assert_close(
        expanded.corrected.quadratic_at(probe),
        (g + torch.einsum("kip,kp->ki", H, probe)) ** 2,
    )

    displacement = torch.randn_like(g)
    predicted_gradient = g + torch.einsum("kip,kp->ki", H, displacement)
    torch.testing.assert_close(
        expanded.quadratic_at(displacement), predicted_gradient.square()
    )
    torch.testing.assert_close(
        expanded.at(displacement), predicted_gradient.abs() + eps
    )


def test_denominator_moment_compress_does_not_transform_state() -> None:
    torch.manual_seed(43)
    beta = 0.7
    moment = DenominatorMoment(beta, eps=1e-8)
    g = torch.randn(2, 3, dtype=torch.float64)
    H = torch.randn(2, 3, 3, dtype=torch.float64)
    first = moment.expand(
        moment.initialize(g),
        AtomGradientObservation(jg=g, gh=H, contributions=1),
        next_step=1,
    )
    state = moment.compress(first, torch.randn_like(g))

    assert isinstance(state, DenominatorMomentState)
    torch.testing.assert_close(state.x, first.raw.x)
    torch.testing.assert_close(state.y, first.raw.y)
    torch.testing.assert_close(state.Z, first.raw.Z)

    g2 = torch.randn_like(g)
    H2 = torch.randn_like(H)
    second = moment.expand(
        state,
        AtomGradientObservation(jg=g2, gh=H2, contributions=1),
        next_step=2,
    )
    current_x = g2.square()
    current_y = torch.einsum("ki,kip->kip", g2, H2)
    current_Z = torch.einsum("kip,kiq->kipq", H2, H2)
    expected_x = beta * first.raw.x + (1.0 - beta) * current_x
    expected_y = beta * first.raw.y + (1.0 - beta) * current_y
    expected_Z = beta * first.raw.Z + (1.0 - beta) * current_Z
    torch.testing.assert_close(second.raw.x, expected_x)
    torch.testing.assert_close(second.raw.y, expected_y)
    torch.testing.assert_close(second.raw.Z, expected_Z)


def test_numerator_moment_compress_does_not_recenter() -> None:
    torch.manual_seed(37)
    beta = 0.7
    moment = NumeratorMoment(beta)
    g1 = torch.randn(2, 3, dtype=torch.float64)
    H1 = torch.randn(2, 3, 3, dtype=torch.float64)
    first = moment.expand(
        moment.initialize(g1),
        AtomGradientObservation(jg=g1, gh=H1, contributions=1),
        next_step=1,
    )
    displacement = torch.randn_like(g1)
    state = moment.compress(first, displacement)

    assert isinstance(state, NumeratorMomentState)
    torch.testing.assert_close(state.m, first.raw.constant)
    torch.testing.assert_close(state.C, first.raw.linear)

    g2 = torch.randn_like(g1)
    H2 = torch.randn_like(H1)
    second = moment.expand(
        state,
        AtomGradientObservation(jg=g2, gh=H2, contributions=1),
        next_step=2,
    )

    expected_m = beta * first.raw.constant + (1.0 - beta) * g2
    expected_C = beta * first.raw.linear + (1.0 - beta) * H2
    torch.testing.assert_close(second.raw.constant, expected_m)
    torch.testing.assert_close(second.raw.linear, expected_C)
    torch.testing.assert_close(
        second.corrected.constant,
        expected_m / (1.0 - beta**2),
    )
    torch.testing.assert_close(
        second.corrected.linear,
        expected_C / (1.0 - beta**2),
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

    system.expand(state, observation, context)

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
