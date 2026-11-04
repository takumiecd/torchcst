"""Memory admission rejects continuing growth and missing evidence independently."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.training.full_step_graph_memory import (
    ENDPOINT_PHASE,
    GATE_VERSION,
    evaluate_lifecycle,
)

MIB = 1024**2


def report(reserved):
    return {
        "total_updates": 128,
        "replay_updates": 120,
        "initial_state": {"p": "initial"},
        "initialization_sha256": "fixture",
        "optimizer_contract": {"lr": 1e-4},
        "data": "fixed",
        "unexpected_fallbacks": 0,
        "peak_allocated_bytes": 252 * MIB,
        "peak_reserved_bytes": max(reserved) * MIB,
        "epochs": [
            {
                "epoch": i,
                "steps": 16,
                "total_updates": 16 * i,
                "state": {str(k): f"hash-{i}-{k}" for k in range(42)},
                "gradients": {str(k): f"grad-{i}-{k}" for k in range(5)},
                "totals": [i * 1.5, i],
                "train": {"loss": 0.1},
                "test": {"loss": 0.2},
                "data_order_sha256": f"order-{i}",
                "end_allocated_bytes": 245 * MIB,
                "end_reserved_bytes": value * MIB,
            }
            for i, value in enumerate(reserved, 1)
        ],
    }


def evaluate(a, b, **kwargs):
    return evaluate_lifecycle(
        a,
        b,
        evidence_scope="prospective",
        baseline_run="baseline",
        candidate_run="candidate",
        endpoint_phase=kwargs.get("endpoint_phase", ENDPOINT_PHASE),
    )


@pytest.mark.parametrize("values", [[442] + [444] * 7, [444] * 8])
def test_initial_reservation_plateau_and_constant_control_are_admitted(values):
    result = evaluate(report([444] * 8), report(values))
    assert result["passed"] is True and result["status"] == "pass"
    assert result["gate_version"] == GATE_VERSION


@pytest.mark.parametrize(
    "change",
    [
        "reserved_growth",
        "allocated_growth",
        "transient_peak",
        "state",
        "gradient",
        "order",
        "fallback",
        "start",
        "phase",
        "updates",
    ],
)
def test_rejects_regression_in_distinct_evidence_dimensions(change):
    a, b = report([444] * 8), report([442] + [444] * 7)
    if change == "reserved_growth":
        b["epochs"][-1]["end_reserved_bytes"] += MIB
        b["peak_reserved_bytes"] += MIB
    elif change == "allocated_growth":
        b["epochs"][-1]["end_allocated_bytes"] += 512
    elif change == "transient_peak":
        b["peak_reserved_bytes"] += 9 * MIB
    elif change == "state":
        b["epochs"][2]["state"]["0"] = "different"
    elif change == "gradient":
        b["epochs"][2]["gradients"]["0"] = "different"
    elif change == "order":
        b["epochs"][2]["data_order_sha256"] = "different"
    elif change == "fallback":
        b["unexpected_fallbacks"] = 1
    elif change == "start":
        b["initial_state"] = {}
    elif change == "phase":
        b["memory_endpoint_phase"] = "after_retirement"
    else:
        b["epochs"][2]["steps"] = 15
    result = evaluate(a, b)
    assert result["status"] == "fail" and result["passed"] is False


def test_three_epochs_are_indeterminate_and_extra_cycles_cannot_select_a_suffix():
    a, b = report([444] * 8), report([444] * 8)
    b["epochs"] = b["epochs"][:3]
    assert evaluate(a, b)["status"] == "indeterminate"
    b["epochs"] *= 3
    assert evaluate(a, b)["status"] == "fail"


def test_missing_phase_or_values_never_pass():
    a, b = report([444] * 8), report([444] * 8)
    assert evaluate(a, b, endpoint_phase="unknown")["status"] == "indeterminate"
    del b["peak_reserved_bytes"]
    assert evaluate(a, b)["status"] == "indeterminate"


def test_existing_sort32_supplement_passes_without_mutating_legacy_false():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/research-history/cuda-linear/owner-window-sort32-results-20261104.json"
    )
    before = path.read_bytes()
    original = json.loads(before)
    data = copy.deepcopy(original)
    life = data["continuation"]["lifecycle8"]
    result = evaluate_lifecycle(
        life["base"],
        life["sort32"],
        evidence_scope="retrospective_supplemental",
        baseline_run="sort32/lifecycle-base",
        candidate_run="sort32/lifecycle-sort32",
        endpoint_phase=ENDPOINT_PHASE,
    )
    assert result["status"] == "pass"
    assert data == original and data["decision"]["long_gates_pass"] is False
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()
