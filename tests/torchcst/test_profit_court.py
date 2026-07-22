"""ProfitCourt accept/reject integration and complete trial restoration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from torchcst.compute import RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import LC_merge
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseMerge, SynapseStore


def _engine(source: torch.Tensor, target: torch.Tensor, *, cost_rate: float):
    store = SynapseStore(
        "rank",
        2,
        2,
        2,
        spec=RepresentationSpec.rank_one(2, 2),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                "rank",
                source,
                target,
                torch.tensor([1.0, 0.5], dtype=torch.float64),
                torch.tensor([20, 21], dtype=torch.int64),
            )
        ]
    )
    module = RankOneLinear(store, 2, 2)
    optimizer = torch.optim.Adam(store.parameters(), lr=0.01)
    optimizer.state[store.w] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([0.25, -0.5], dtype=torch.float64),
        "exp_avg_sq": torch.tensor([0.1, 0.2], dtype=torch.float64),
    }
    policy = LC_merge(
        event_interval=1,
        birth_budget=1,
        similarity_threshold=0.0,
        cost_rate=cost_rate,
    )
    engine = StructuralEngine(
        {"rank": store},
        policy,
        modules={"rank": module},
        optimizer=optimizer,
        seed=13,
    )
    return store, module, optimizer, engine


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            _assert_equal(one, two)
    else:
        assert left == right


def test_nearly_parallel_merge_is_accepted_and_objective_is_read_twice() -> None:
    angle = 0.01
    source = torch.tensor(
        [[1.0, 0.0], [torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))]],
        dtype=torch.float64,
    )
    target = source.clone()
    store, module, _, engine = _engine(source, target, cost_rate=0.01)
    desired = module.dense_weight().detach().clone()
    calls = 0

    def objective() -> float:
        nonlocal calls
        calls += 1
        return float((module.dense_weight() - desired).square().sum().detach())

    ops = engine.step(objective)

    assert any(isinstance(op, SynapseMerge) for op in ops)
    assert store.view().ids.numel() == 1
    assert calls == 2
    assert len(engine.op_log()) == 1


def test_orthogonal_merge_reject_restores_store_optimizer_court_rng_registry_and_log() -> None:
    source = torch.eye(2, dtype=torch.float64)
    store, module, optimizer, engine = _engine(source, source.clone(), cost_rate=0.0)
    desired = module.dense_weight().detach().clone()
    live_id = int(store.live_ids()[0])
    engine.policy.synapse_retention_rule._strike_state[live_id] = 1
    before = {
        "store": deepcopy(store.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "strikes": engine.policy.synapse_retention_rule.state_dict(),
        "rng": engine.rng.get_state().clone(),
        "registry": engine.registry.state_dict(),
        "op_log": engine.op_log(),
    }
    calls = 0

    def objective() -> float:
        nonlocal calls
        calls += 1
        torch.rand((), generator=engine.rng)
        return float((module.dense_weight() - desired).square().sum().detach())

    ops = engine.step(objective)

    assert ops == ()
    assert calls == 2
    # The rejected trial is bit-restored; the surrounding scheduled event then
    # advances the age clock once, as it does for every event (including frozen
    # and no-op events).  Normalize only that post-transaction clock effect.
    expected_store = deepcopy(before["store"])
    expected_store["_extra_state"]["followers"]["followers"][0]["values"] += 1
    _assert_equal(store.state_dict(), expected_store)
    _assert_equal(optimizer.state_dict(), before["optimizer"])
    _assert_equal(
        engine.policy.synapse_retention_rule.state_dict(), before["strikes"]
    )
    assert torch.equal(engine.rng.get_state(), before["rng"])
    _assert_equal(engine.registry.state_dict(), before["registry"])
    assert engine.op_log() == before["op_log"]
    assert engine.clock.update_step == engine.clock.event_index == 1
