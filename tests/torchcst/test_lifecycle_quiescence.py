"""Scripted, non-statistical lifecycle quiescence tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from torchcst.engine import StructuralEngine
from torchcst.policy import (
    BudgetDistributor,
    Cadence,
    Clock,
    EvenBudgetDistributor,
    OpProposer,
    PeriodicCadence,
    QuotaRegime,
    RetentionCourt,
    SynapseLifecycle,
    UniformEntryBirth,
)
from tests.torchcst._recipes import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _birth_count(ops) -> int:
    return sum(op.w.numel() for op in ops if isinstance(op, SynapseBirth))


def _death_ids(ops) -> list[int]:
    return [
        entity_id
        for op in ops
        if isinstance(op, SynapseDeath)
        for entity_id in op.ids.tolist()
    ]


def _set_scripted_mass(store: SynapseStore, weak_lineage: int) -> None:
    slots = store._slots.live_slots
    with torch.no_grad():
        store.w.index_fill_(0, slots.to(store.w.device), 10.0)
        weak_slots = slots[store.lineage.values.index_select(0, slots) == weak_lineage]
        if weak_slots.numel():
            store.w.index_fill_(0, weak_slots.to(store.w.device), 0.1)


def test_lifecycle_prunes_after_immunity_then_becomes_quiescent() -> None:
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=8),
    )
    engine = StructuralEngine(
        {"entry": store},
        LC(
            event_interval=1,
            birth_end_event=6,
            birth_budget=2,
            freeze_event=10,
            bounds_in=1,
            bounds_out=8,
        ),
        seed=4,
    )

    first = engine.step()
    weak_lineage = first[0].lineage[0].item()
    _set_scripted_mass(store, weak_lineage)
    events = [first]
    for _ in range(9):
        events.append(engine.step())
        _set_scripted_mass(store, weak_lineage)

    assert all(not _death_ids(ops) for ops in events[:3])
    assert not _death_ids(events[3])  # first eligible strike
    assert len(_death_ids(events[4])) == 1  # second consecutive strike
    assert weak_lineage in {
        lineage for site, lineage in engine.registry.snapshot() if site == "entry"
    }
    assert _birth_count(events[5]) == 0  # retired free coordinate is not reborn
    assert all(len(ops) == 0 for ops in events[5:])
    assert _birth_count(events[6]) == 0  # event 7 is outside the birth window


class _BadCourt:
    immunity_events = 3

    def decide(self, view, ages, clock):
        return (SynapseDeath(view.site, view.ids[:1]),)


def test_engine_runtime_rejects_any_court_death_during_immunity() -> None:
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.zeros(1, 1, dtype=torch.int64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    method = SynapseLifecycle(
        # budget=0 below means this birth rule is never actually called; a
        # lifecycle still must declare *some* birth or absorb rule to build.
        birth_factory=lambda lam: UniformEntryBirth(bounds_in=1, bounds_out=1),
        prune_factory=lambda: _BadCourt(),
        priceable=False,
        label="bad-court",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        distributor=EvenBudgetDistributor(),
    )

    engine = StructuralEngine({"entry": store}, root, seed=0)
    engine.begin_update()
    with pytest.raises(RuntimeError, match="immune"):
        engine.step()

    # A rejected policy event consumes its clock tick, but the failed update
    # must not leak capture state into the next training update.
    assert engine.clock == Clock(update_step=1, event_index=1)
    assert engine.update_id is None
    assert not engine.capture_active


def test_loss_blind_policy_signatures() -> None:
    components = (Cadence, OpProposer, BudgetDistributor, RetentionCourt)
    forbidden = ("loss", "objective")
    for component in components:
        for name, member in inspect.getmembers(component):
            if name.startswith("_") or not callable(member):
                continue
            parameters = inspect.signature(member).parameters
            assert all(
                word not in parameter.lower()
                for parameter in parameters
                for word in forbidden
            )


def test_clock_has_only_the_minimal_two_fields() -> None:
    assert tuple(Clock.__dataclass_fields__) == ("update_step", "event_index")
