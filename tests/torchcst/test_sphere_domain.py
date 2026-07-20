"""Sphere coordinate and optimizer-gauge contracts."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import LC
from torchcst.representation import IntegerGrid, RepresentationSpec, Sphere
from torchcst.storage import SynapseBirth, SynapseStore


def test_sphere_sample_retract_and_tangent_projection() -> None:
    sphere = Sphere(3)
    rng = torch.Generator().manual_seed(7)
    sample = sphere.sample(128, rng)
    torch.testing.assert_close(sample.norm(dim=1), torch.ones(128))

    coords = sphere.retract(torch.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]]))
    grad = torch.tensor([[2.0, -1.0, 3.0], [4.0, 5.0, 6.0]])
    projected = sphere.project_grad(coords, grad)
    torch.testing.assert_close((projected * coords).sum(1), torch.zeros(2))
    with pytest.raises(ValueError, match="zero"):
        sphere.retract(torch.zeros(1, 3))


def test_sphere_project_state_changes_signed_moments_only() -> None:
    sphere = Sphere(2)
    coords = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    momentum = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    exp_avg_sq = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    state = {
        "momentum_buffer": momentum,
        "exp_avg_sq": exp_avg_sq,
        "unrelated": torch.ones(3),
    }
    expected_sq = exp_avg_sq.clone()

    sphere.project_state(coords, state)

    torch.testing.assert_close((momentum * coords).sum(1), torch.zeros(2))
    assert torch.equal(exp_avg_sq, expected_sq)


def test_integer_grid_sampling_and_lineage_encoding() -> None:
    domain = IntegerGrid((2, 3))
    sample = domain.sample(50, torch.Generator().manual_seed(2))
    domain.validate_birth(sample)
    expected = sample[:, 0] * 3 + sample[:, 1]
    assert torch.equal(domain.lineage_key(sample), expected)
    with pytest.raises(ValueError, match="bounds"):
        IntegerGrid().sample(1, torch.Generator())


def test_engine_restores_sphere_gauge_and_projects_sgd_momentum() -> None:
    store = SynapseStore(
        "rank", 2, 2, 2, spec=RepresentationSpec.rank_one(2, 2)
    )
    store.apply(
        [
            SynapseBirth(
                "rank",
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([1.0, -0.5]),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    module = RankOneLinear(store, 2, 2)
    optimizer = torch.optim.SGD(store.parameters(), lr=0.2, momentum=0.9)
    engine = StructuralEngine(
        {"rank": store}, LC(event_interval=100, birth_budget=0), optimizer=optimizer
    )

    for _ in range(4):
        optimizer.zero_grad()
        module(torch.tensor([[1.5, -0.75]])).square().sum().backward()
        optimizer.step()
        engine.step()

    view = store.view()
    torch.testing.assert_close(view.s.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0)
    slots = store._slots.live_slots.to(store.s.device)
    for coordinate in (store.s, store.t):
        momentum = optimizer.state[coordinate]["momentum_buffer"].index_select(0, slots)
        live = coordinate.index_select(0, slots)
        torch.testing.assert_close(
            (momentum * live).sum(1), torch.zeros(2), atol=1e-6, rtol=0
        )
