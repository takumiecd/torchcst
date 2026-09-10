"""Projected-visible local Adam equations and public optimizer contract."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.test_local_optimizer import model, step
from torchcst import (
    AdamWConfig,
    CSTLocalAdam,
    CSTLocalVisibleAdam,
    LocalVisibleAdamConfig,
)
from torchcst.optim.atom_grad import AtomGradientObservation
from torchcst.optim.moments.projected_visible import (
    ProjectedVisibleSecondMoment,
    ProjectedVisibleSecondMomentState,
)


def context(current_j, previous_j, point):
    def cross(left, right, *, local):
        assert local
        other = current_j if left is right else previous_j
        return current_j.transpose(-1, -2) @ other

    return SimpleNamespace(
        current_point=point,
        geometry=SimpleNamespace(cross=cross, visible_shape=(current_j.shape[1], 1)),
    )


def congruence_solve(matrix, value):
    left = torch.linalg.solve(matrix, value)
    return torch.linalg.solve(matrix, left.transpose(-1, -2)).transpose(-1, -2)


def test_projected_visible_observation_and_geometric_mean_match_oracle():
    generator = torch.Generator().manual_seed(91)
    j = torch.randn(3, 7, 4, generator=generator, dtype=torch.float64)
    visible_gradient = torch.randn(7, generator=generator, dtype=torch.float64)
    evidence = torch.einsum(
        "knp,n,knq->kpq", j, visible_gradient.square(), j
    )
    point = torch.zeros(3, 4, dtype=torch.float64)
    ctx = context(j, j, point)
    component = ProjectedVisibleSecondMoment(
        0.8, eps=1e-6, damping=1e-3, rtol=1e-10
    )
    result = component.expand(
        component.initialize(ctx),
        AtomGradientObservation(atom_square=evidence),
        ctx,
        next_step=1,
    )

    gram = j.transpose(-1, -2) @ j
    regularized = gram + 1e-3 * torch.eye(4, dtype=torch.float64)
    raw = 0.2 * evidence
    torch.testing.assert_close(
        result.pending_state.coefficient,
        congruence_solve(regularized, raw),
    )
    factor = torch.linalg.cholesky(regularized)
    normalized = torch.linalg.solve_triangular(factor, evidence, upper=False)
    normalized = torch.linalg.solve_triangular(
        factor, normalized.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    values, vectors = torch.linalg.eigh(normalized)
    root = (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(
        -1, -2
    )
    expected_metric = factor @ (
        root + 1e-6 * torch.eye(4, dtype=torch.float64)
    ) @ factor.transpose(-1, -2)
    torch.testing.assert_close(result.metric.blocks, expected_metric)


def test_geometric_mean_is_exact_for_the_projected_ambient_representative():
    generator = torch.Generator().manual_seed(19)
    j = torch.randn(1, 6, 3, generator=generator, dtype=torch.float64)
    visible_second = torch.rand(6, generator=generator, dtype=torch.float64) + 0.2
    evidence = torch.einsum("knp,n,knq->kpq", j, visible_second, j)
    point = torch.zeros(1, 3, dtype=torch.float64)
    ctx = context(j, j, point)
    component = ProjectedVisibleSecondMoment(
        0.0, eps=1e-6, damping=0.0, rtol=1e-10
    )
    result = component.expand(
        component.initialize(ctx),
        AtomGradientObservation(atom_square=evidence),
        ctx,
        next_step=1,
    )

    gram = j.transpose(-1, -2) @ j
    coefficient = congruence_solve(gram, evidence)
    ambient = j @ coefficient @ j.transpose(-1, -2)
    values, vectors = torch.linalg.eigh(ambient)
    ambient_root = (
        vectors * values.clamp_min(0).sqrt().unsqueeze(-2)
    ) @ vectors.transpose(-1, -2)
    exact_for_representative = (
        j.transpose(-1, -2) @ ambient_root @ j + 1e-6 * gram
    )
    torch.testing.assert_close(
        result.metric.blocks, exact_for_representative, atol=1e-9, rtol=1e-8
    )


def test_projected_visible_transport_is_the_old_operator_seen_by_new_jacobian():
    generator = torch.Generator().manual_seed(52)
    old_j = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    new_j = old_j + 0.1 * torch.randn(
        2, 6, 3, generator=generator, dtype=torch.float64
    )
    old_point = torch.zeros(2, 3, dtype=torch.float64)
    old_context = context(old_j, old_j, old_point)
    component = ProjectedVisibleSecondMoment(
        0.9, eps=1e-8, damping=1e-3, rtol=1e-10
    )
    evidence = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
    evidence = evidence @ evidence.transpose(-1, -2)
    first = component.expand(
        component.initialize(old_context),
        AtomGradientObservation(atom_square=evidence),
        old_context,
        next_step=1,
    )

    new_point = torch.ones(2, 3, dtype=torch.float64)
    new_context = context(new_j, old_j, new_point)
    second = component.expand(
        first.pending_state,
        AtomGradientObservation(atom_square=torch.zeros_like(evidence)),
        new_context,
        next_step=2,
    )
    cross = new_j.transpose(-1, -2) @ old_j
    transported = 0.9 * (
        cross
        @ first.pending_state.coefficient
        @ cross.transpose(-1, -2)
    )
    new_gram = new_j.transpose(-1, -2) @ new_j
    regularized = new_gram + 1e-3 * torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(
        second.pending_state.coefficient,
        congruence_solve(regularized, transported),
    )


def test_public_visible_optimizer_requests_atom_square_and_resumes():
    original = model(torch.float64)
    config = LocalVisibleAdamConfig(second_moment_damping=1e-3)
    optimizer = CSTLocalVisibleAdam(original, cst=config, dense=AdamWConfig())
    request = optimizer._sites[0].moments.observation_request
    assert request.jg and request.atom_square
    assert not request.row_square and not request.column_square
    for _ in range(2):
        step(original, optimizer)

    state = optimizer._sites[0].state.second
    assert isinstance(state, ProjectedVisibleSecondMomentState)
    assert state.coefficient.shape == (
        original[0].atoms.p.shape[0],
        original[0].atoms.p.shape[1],
        original[0].atoms.p.shape[1],
    )
    assert not hasattr(state, "basis")

    restored = model(torch.float64)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    resumed = CSTLocalVisibleAdam(restored, cst=config, dense=AdamWConfig())
    resumed.load_state_dict(optimizer.state_dict())
    step(original, optimizer)
    step(restored, resumed)
    for actual, expected in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    with pytest.raises(ValueError, match="algorithm"):
        CSTLocalAdam(model(torch.float64), dense=AdamWConfig()).load_state_dict(
            optimizer.state_dict()
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"second_moment_damping": 0},
        {"second_moment_damping": float("nan")},
        {"update_damping": 0},
        {"solve_rtol": 1},
    ],
)
def test_visible_config_rejects_invalid_regularization(kwargs):
    with pytest.raises(ValueError):
        LocalVisibleAdamConfig(**kwargs)
