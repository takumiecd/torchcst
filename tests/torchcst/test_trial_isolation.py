"""Profit trials are impossible to reach from ordinary (non-priced) trees."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch

from torchcst.compute import EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    Clock,
    PeriodicCadence,
    ProfitCourt,
    QuotaRegime,
    TrialTransaction,
    cRigL,
    cSET,
)
from tests.torchcst._recipes import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _cset_policy(*, event_interval: int, birth_budget: int):
    return QuotaRegime(
        budget=birth_budget,
        method=cSET(),
        cadence=PeriodicCadence(event_interval=event_interval),
    )


def _crigl_policy(
    *, event_interval: int, birth_budget: int, observe_window: int, pool_size: int
):
    return QuotaRegime(
        budget=birth_budget,
        method=cRigL(pool_size=pool_size),
        cadence=PeriodicCadence(
            event_interval=event_interval, observe_window=observe_window
        ),
    )


def _assert_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            _assert_state_equal(one, two)
    else:
        assert left == right


def _entry_engine(policy) -> StructuralEngine:
    store = SynapseStore(
        "entry",
        1,
        1,
        2,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([1.0, 2.0]),
                torch.tensor([0, 3], dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 2, 2)
    # An unbuilt root has no .requires (that is only a bound RuntimeTree
    # property, policy/tree.py); passing the module unconditionally is
    # harmless whether or not the eventual bound tree ends up requiring
    # capture.
    return StructuralEngine({"entry": store}, policy, modules={"entry": module})


def test_objective_is_rejected_when_profit_capability_is_absent() -> None:
    engine = _entry_engine(LC(event_interval=1, birth_budget=0))
    with pytest.raises(RuntimeError, match="tree has no profit court"):
        engine.step(lambda: 0.0)
    assert engine.clock.update_step == 0


# (test_lc_merge_requires_objective_when_a_merge_is_proposed removed: LC_merge
# and SynapseMerge-as-a-tree-action were retired outright in Phase 2 S4e
# (docs/policy-tree-phase2.md "消すもの") -- the tree vocabulary
# (SynapseLifecycle's birth/prune/absorb slots) has no merge action at all,
# so there is no tree-native way to reconstruct "a merge is proposed" as a
# premise. LC_merge's historical value is preserved in git history.)


@pytest.mark.parametrize(
    "policy",
    [
        LC(event_interval=1, birth_budget=0),
        _cset_policy(event_interval=1, birth_budget=0),
        _crigl_policy(event_interval=1, birth_budget=0, observe_window=1, pool_size=2),
    ],
)
def test_normal_trees_have_no_trial_state_or_objective_wiring(policy) -> None:
    engine = _entry_engine(policy)

    assert engine._tree.profit is None
    assert not hasattr(engine, "objective")
    assert not hasattr(engine, "_trial_transaction")
    assert not hasattr(engine, "_trial_session")
    assert isinstance(engine.step(), tuple)


def test_trial_transaction_restores_instrument_followers_clock_and_rng_bits() -> None:
    base = _crigl_policy(
        event_interval=1,
        birth_budget=0,
        observe_window=1,
        pool_size=2,
    )
    engine = _entry_engine(replace(base, profit=ProfitCourt()))
    store = engine.synapse_stores["entry"]
    instrument = engine.instrument("entry", "candidate_field")
    instrument.reconcile(store.view())
    before = {
        "store": deepcopy(store.state_dict()),
        "instrument": deepcopy(instrument.state_dict()),
        "registry": deepcopy(engine.registry.state_dict()),
        "clock": engine.clock,
        "rng": engine.rng.get_state().clone(),
        "op_log": engine.op_log(),
    }
    transaction = TrialTransaction(engine)

    with torch.no_grad():
        store.w.add_(17.0)
    store.age.values.add_(4)
    store.lineage.values.add_(9)
    store._version += 3
    instrument._scores.add_(5.0)
    engine.registry.retire("entry", 99)
    engine.clock = Clock(7, 3)
    engine._op_log.append(
        (3, SynapseDeath("entry", store.live_ids()[:1]))
    )
    torch.rand((), generator=engine.rng)

    transaction.rollback()

    _assert_state_equal(store.state_dict(), before["store"])
    _assert_state_equal(instrument.state_dict(), before["instrument"])
    _assert_state_equal(engine.registry.state_dict(), before["registry"])
    assert engine.clock == before["clock"]
    assert torch.equal(engine.rng.get_state(), before["rng"])
    assert engine.op_log() == before["op_log"]
