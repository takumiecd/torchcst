import torch

from torchcst.optim import (
    AtomGradientObservation,
    DenominatorMoment,
    DenominatorMomentState,
    ExpandedDenominatorMoment,
    ExpandedNumeratorMoment,
    NumeratorMoment,
    NumeratorMomentState,
)


def test_numerator_moment_matches_affine_ema_and_bias_correction() -> None:
    beta = 0.8
    moment = NumeratorMoment(beta)
    g = torch.randn(2, 3, dtype=torch.float64)
    hessian = torch.randn(2, 3, 3, dtype=torch.float64)
    observation = AtomGradientObservation(jg=g, gh=hessian, contributions=1)

    expanded = moment.expand(moment.initialize(g), observation, next_step=1)

    assert isinstance(expanded, ExpandedNumeratorMoment)
    torch.testing.assert_close(expanded.raw.constant, (1.0 - beta) * g)
    torch.testing.assert_close(expanded.raw.linear, (1.0 - beta) * hessian)
    torch.testing.assert_close(expanded.corrected.constant, g)
    torch.testing.assert_close(expanded.corrected.linear, hessian)


def test_denominator_moment_matches_squared_affine_gradient() -> None:
    beta = 0.8
    eps = 1e-6
    moment = DenominatorMoment(beta, eps=eps)
    g = torch.randn(2, 3, dtype=torch.float64)
    hessian = torch.randn(2, 3, 3, dtype=torch.float64)
    observation = AtomGradientObservation(jg=g, gh=hessian, contributions=1)

    expanded = moment.expand(moment.initialize(g), observation, next_step=1)

    assert isinstance(expanded, ExpandedDenominatorMoment)
    probe = torch.randn_like(g)
    predicted = g + torch.einsum("kip,kp->ki", hessian, probe)
    torch.testing.assert_close(expanded.quadratic_at(probe), predicted.square())
    torch.testing.assert_close(expanded.at(probe), predicted.abs() + eps)


def test_denominator_compress_commits_pending_state_without_recentring() -> None:
    moment = DenominatorMoment(0.7, eps=1e-8)
    g = torch.randn(2, 3, dtype=torch.float64)
    hessian = torch.randn(2, 3, 3, dtype=torch.float64)
    expanded = moment.expand(
        moment.initialize(g),
        AtomGradientObservation(jg=g, gh=hessian, contributions=1),
        next_step=1,
    )

    state = moment.compress(expanded, torch.randn_like(g))

    assert isinstance(state, DenominatorMomentState)
    torch.testing.assert_close(state.x, expanded.raw.x)
    torch.testing.assert_close(state.y, expanded.raw.y)
    torch.testing.assert_close(state.Z, expanded.raw.Z)


def test_numerator_compress_commits_pending_state_without_recentring() -> None:
    moment = NumeratorMoment(0.7)
    g = torch.randn(2, 3, dtype=torch.float64)
    hessian = torch.randn(2, 3, 3, dtype=torch.float64)
    expanded = moment.expand(
        moment.initialize(g),
        AtomGradientObservation(jg=g, gh=hessian, contributions=1),
        next_step=1,
    )

    state = moment.compress(expanded, torch.randn_like(g))

    assert isinstance(state, NumeratorMomentState)
    torch.testing.assert_close(state.m, expanded.raw.constant)
    torch.testing.assert_close(state.C, expanded.raw.linear)
