"""Dense visible Adam moments and atom-local update contract."""

import copy

import pytest
import torch

from tests.test_local_optimizer import model, step
from torchcst import (
    AdamWConfig,
    CSTDenseVisibleAdam,
    DenseVisibleAdamConfig,
)
from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry
from torchcst.optim.atom_grad import AtomGradientObservation
from torchcst.optim.moments import MomentContext, MomentSystem
from torchcst.optim.moments.dense_visible import (
    DenseVisibleFirstMoment,
    DenseVisibleFirstMomentState,
    DenseVisibleSecondMoment,
    DenseVisibleSecondMomentState,
)


@pytest.mark.parametrize("factored", [False, True])
def test_dense_visible_moments_match_exact_local_adam_oracle(factored):
    site = model(torch.float64)[0]
    derivatives = site.cst_derivatives()
    geometry = AtomLocalTangentGeometry(derivatives, factored=factored)
    point = geometry.current_point()
    context = MomentContext(geometry, point)
    generator = torch.Generator().manual_seed(84)
    gradient = torch.randn(
        geometry.visible_shape, generator=generator, dtype=torch.float64
    )
    system = MomentSystem(
        first=DenseVisibleFirstMoment(0.9),
        second=DenseVisibleSecondMoment(
            0.99, eps=1e-7, row_chunk=1, atom_chunk=1
        ),
    )

    expanded = system.expand(
        system.initialize(context),
        AtomGradientObservation(visible_gradient=gradient),
        context,
    )

    expected_first = geometry.pullback(gradient, point=point)
    columns = geometry.row_columns(
        point, 0, gradient.shape[0], 0, point.shape[0]
    )
    expected_metric = torch.einsum(
        "krip,ri,kriq->kpq",
        columns,
        gradient.abs() + 1e-7,
        columns,
    )
    torch.testing.assert_close(expanded.first.corrected.constant, expected_first)
    torch.testing.assert_close(expanded.second.metric.blocks, expected_metric)


def test_public_dense_visible_optimizer_keeps_dense_m_v_and_resumes():
    original = model(torch.float64)
    config = DenseVisibleAdamConfig()
    optimizer = CSTDenseVisibleAdam(original, cst=config, dense=AdamWConfig())
    request = optimizer._sites[0].moments.observation_request
    assert request.visible_gradient
    assert not request.jg and not request.gh and not request.atom_square
    for _ in range(2):
        step(original, optimizer)

    site = optimizer._sites[0]
    first, second = site.state.first, site.state.second
    assert isinstance(first, DenseVisibleFirstMomentState)
    assert isinstance(second, DenseVisibleSecondMomentState)
    assert first.value.shape == (
        site.module.out_features,
        site.module.in_features,
    )
    assert second.value.shape == first.value.shape

    restored = model(torch.float64)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    resumed = CSTDenseVisibleAdam(restored, cst=config, dense=AdamWConfig())
    resumed.load_state_dict(optimizer.state_dict())
    step(original, optimizer)
    step(restored, resumed)
    for actual, expected in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": 0},
        {"update_damping": 0},
        {"update_damping": float("nan")},
        {"solve_rtol": 1},
    ],
)
def test_dense_visible_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DenseVisibleAdamConfig(**kwargs)
