"""Aggregate economy audit and one-way engine subscription contracts."""

from __future__ import annotations

import json

import pytest
import torch

from torchcst.audit import AuditRecord, EconomyAudit
from torchcst.engine import StructuralEngine
from torchcst.lab import Ledger
from torchcst.policy import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _run(*, audited: bool):
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=6),
    )
    audit = EconomyAudit(immunity_events=3) if audited else None
    engine = StructuralEngine(
        {"entry": store},
        LC(
            event_interval=1,
            birth_end_event=3,
            birth_budget=2,
            freeze_event=5,
            bounds_in=1,
            bounds_out=6,
            initial_weight=1.0,
        ),
        seed=17,
        audit_subscribers=[] if audit is None else [audit],
    )
    event_ops = [engine.step() for _ in range(6)]
    canonical = json.dumps(
        [
            {"event_index": event, "op": Ledger.serialize_ops((op,))[0]}
            for event, op in engine.op_log()
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return audit, event_ops, canonical


def test_lc_economy_matches_op_log_and_detects_quiescence() -> None:
    audit, events, _ = _run(audited=True)
    assert audit is not None

    births = [
        sum(int(op.w.numel()) for op in ops if isinstance(op, SynapseBirth))
        for ops in events
    ]
    deaths = [
        sum(int(op.ids.numel()) for op in ops if isinstance(op, SynapseDeath))
        for ops in events
    ]
    assert audit.k_series("entry") == (2, 4, 6, 6, 6, 6)
    assert audit.churn == tuple(birth + death for birth, death in zip(births, deaths))
    assert audit.thrash_rate == 0.0
    assert audit.last_structural_event == 3
    assert audit.quiescence_step == 3
    assert audit.rent_margins["entry"][1].minimum == pytest.approx(1 / 0.3)


def test_audit_subscription_cannot_change_structural_op_log() -> None:
    _, _, without_audit = _run(audited=False)
    audit, _, with_audit = _run(audited=True)

    assert audit is not None and len(audit.records) == 6
    assert with_audit == without_audit


def test_user_loss_readings_are_audit_only() -> None:
    audit = EconomyAudit()
    audit.record_loss(10, 1.25)
    audit.record_loss(11, 0.75)

    assert [(row.update_step, row.value) for row in audit.losses] == [
        (10, 1.25),
        (11, 0.75),
    ]


def test_default_maturity_is_immunity_plus_strikes_minus_one() -> None:
    audit = EconomyAudit(immunity_events=3, strikes=2)
    assert audit.maturity_events == 4


def test_default_maturity_falls_back_to_immunity_events_without_strikes() -> None:
    audit = EconomyAudit(immunity_events=3)
    assert audit.maturity_events == 3


def test_earliest_legal_rent_prune_is_not_thrash() -> None:
    # immunity 3 + 2 consecutive strikes: the earliest legal rent-court death
    # has age immunity_events + strikes - 1 = 4, and must not read as thrash.
    audit = EconomyAudit(immunity_events=3, strikes=2)
    death = SynapseDeath("entry", torch.tensor([4], dtype=torch.int64))
    audit.push(
        AuditRecord(
            event_index=1,
            applied_ops={"entry": (death,)},
            live_counts={"entry": 0},
            mass_snapshots={"entry": torch.zeros(0)},
            prune_ages={"entry": torch.tensor([4], dtype=torch.int64)},
            rent_thresholds={"entry": 0.3},
        )
    )

    assert audit.thrash_count == 0
    assert audit.immune_prunes == 0


def test_immune_prunes_is_always_zero_for_an_engine_run() -> None:
    audit, _, _ = _run(audited=True)
    assert audit is not None
    assert audit.immune_prunes == 0


def test_thrash_denominator_is_all_prunes() -> None:
    # Preserve pre-step-10 semantics explicitly: this test was written when
    # maturity_events defaulted to immunity_events + 2 (= 5 here).  The new
    # default is immunity_events + strikes - 1, so pin the old value directly.
    audit = EconomyAudit(immunity_events=3, maturity_events=5)
    death = SynapseDeath("entry", torch.tensor([4, 9], dtype=torch.int64))
    audit.push(
        AuditRecord(
            event_index=1,
            applied_ops={"entry": (death,)},
            live_counts={"entry": 0},
            mass_snapshots={"entry": torch.zeros(0)},
            prune_ages={"entry": torch.tensor([4, 5], dtype=torch.int64)},
            rent_thresholds={"entry": 0.3},
        )
    )

    assert audit.prune_count == 2
    assert audit.thrash_count == 1
    assert audit.thrash_rate == 0.5
