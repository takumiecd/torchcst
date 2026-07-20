"""Structural replay and canonical ledger acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cstf.engine import StructuralEngine
from cstf.lab import Ledger, RngStreams
from cstf.policy import LC
from cstf.representation import RepresentationSpec
from cstf.storage import SynapseBirth, SynapseDeath, SynapseStore


def run_structure(root_seed: int, *, constructor_seed: int = 0):
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=24),
    )
    streams = RngStreams(root_seed)
    engine = StructuralEngine(
        {"entry": store},
        LC(
            event_interval=1,
            birth_end_event=4,
            birth_budget=3,
            freeze_event=7,
            bounds_in=1,
            bounds_out=24,
            initial_weight=1.0,
        ),
        seed=constructor_seed,
        rng=streams.get("proposal"),
    )
    for _ in range(7):
        engine.step()
    canonical = [
        {"event_index": event_index, "op": Ledger.serialize_ops((op,))[0]}
        for event_index, op in engine.op_log()
    ]
    return engine, json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def test_independent_rebuilds_have_bit_identical_structural_op_logs() -> None:
    first_engine, first = run_structure(2026, constructor_seed=1)
    second_engine, second = run_structure(2026, constructor_seed=999)

    assert first_engine.rng is not second_engine.rng
    assert first == second


def test_structural_op_log_is_sensitive_to_root_seed() -> None:
    _, first = run_structure(2026)
    _, second = run_structure(2027)

    assert first != second


def test_engine_uses_injected_generator_instance_instead_of_seed() -> None:
    streams = RngStreams(3)
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    engine = StructuralEngine(
        {"entry": store},
        LC(event_interval=1, birth_end_event=1, freeze_event=2),
        seed=999,
        rng=streams.get("proposal"),
    )

    assert engine.rng is streams.get("proposal")


def test_ledger_roundtrip_counts_provenance_and_verifies_hash(tmp_path: Path) -> None:
    ledger = Ledger()
    birth = SynapseBirth(
        "entry",
        s=torch.tensor([[0], [1]], dtype=torch.int64),
        t=torch.tensor([[2], [3]], dtype=torch.int64),
        w=torch.tensor([0.25, -0.5], dtype=torch.float32),
        lineage=torch.tensor([7, 8], dtype=torch.int64),
    )
    ledger.record_event(1, (birth,))
    ledger.record_event(
        2,
        (SynapseDeath("entry", torch.tensor([3], dtype=torch.int64)),),
    )
    provenance = ledger.provenance((Path(__file__),))
    path = tmp_path / "ledger.json"

    digest = ledger.to_json(path)
    loaded = Ledger.load(path)

    assert loaded.sha256 == digest
    assert loaded.events == ledger.events
    assert loaded.counters == {"entry": {"birth": 2, "death": 1}}
    assert loaded.provenance_hashes == provenance
    tensor = loaded.events[0]["ops"][0]["w"]
    assert tensor["dtype"] == "torch.float32"
    assert tensor["shape"] == [2]
    assert tensor["data"] == [0.25, -0.5]
    assert loaded.events[1]["ops"][0] == {
        "ids": {"data": [3], "dtype": "torch.int64", "shape": [1]},
        "op": "SynapseDeath",
        "site": "entry",
    }


def test_ledger_rejects_a_one_byte_payload_change(tmp_path: Path) -> None:
    ledger = Ledger()
    ledger.record_event(1, ())
    path = tmp_path / "ledger.json"
    ledger.to_json(path)
    raw = path.read_bytes()
    assert b'"event_index":1' in raw
    path.write_bytes(raw.replace(b'"event_index":1', b'"event_index":2', 1))

    with pytest.raises(ValueError, match="sha256"):
        Ledger.load(path)


def test_ledger_rejects_noncanonical_one_byte_whitespace_change(tmp_path: Path) -> None:
    ledger = Ledger()
    path = tmp_path / "ledger.json"
    ledger.to_json(path)
    path.write_bytes(path.read_bytes()[:-1] + b" \n")

    with pytest.raises(ValueError, match="canonical"):
        Ledger.load(path)
