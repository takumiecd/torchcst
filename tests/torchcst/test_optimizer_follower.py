"""Optimizer state follows physical slot reuse and capacity growth."""

from __future__ import annotations

import torch

from torchcst.engine import StructuralEngine
from torchcst.optim import OptimizerStateFollower
from torchcst.policy import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _birth(store: SynapseStore, count: int, lineage_start: int) -> SynapseBirth:
    source = torch.zeros(count, 3)
    target = torch.zeros(count, 2)
    source[:, 0] = 1.0
    target[:, 0] = 1.0
    return SynapseBirth(
        store.site,
        source,
        target,
        torch.linspace(0.25, 0.25 * count, count),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def test_adam_state_follows_slot_reuse_and_capacity_growth() -> None:
    store = SynapseStore(
        "rank",
        3,
        2,
        capacity=2,
        spec=RepresentationSpec.rank_one(3, 2),
    )
    store.apply([_birth(store, 2, 0)])
    optimizer = torch.optim.Adam(store.parameters(), lr=1.0e-2)
    engine = StructuralEngine(
        {store.site: store},
        LC(event_interval=100, birth_budget=0),
        optimizer=optimizer,
    )

    assert isinstance(engine._optimizer_followers[store.site], OptimizerStateFollower)
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            (update + 1) * store.w.sum()
            + store.s[:, 0].sum()
            + 2.0 * store.t[:, 0].sum()
        )
        loss.backward()
        optimizer.step()

    survivor = 1
    preserved = {
        parameter: {
            name: value[survivor].clone()
            for name, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        }
        for parameter in (store.s, store.t, store.w)
    }
    old_ids = store.live_ids()
    store.apply(
        [
            SynapseDeath(store.site, old_ids[:1]),
            _birth(store, 1, 2),
        ]
    )

    reused_slot = 0
    for parameter in (store.s, store.t, store.w):
        state = optimizer.state[parameter]
        for name in ("exp_avg", "exp_avg_sq"):
            assert bool(torch.count_nonzero(state[name][reused_slot]) == 0)
            torch.testing.assert_close(state[name][survivor], preserved[parameter][name])

    store.apply([_birth(store, 2, 3)])

    assert store.capacity == 4
    follower = engine._optimizer_followers[store.site]
    assert follower.capacity == store.capacity
    for parameter in (store.s, store.t, store.w):
        state = optimizer.state[parameter]
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].shape[0] == store.capacity
            torch.testing.assert_close(state[name][survivor], preserved[parameter][name])
            assert bool(torch.count_nonzero(state[name][[0, 2, 3]]) == 0)
