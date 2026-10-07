"""Synthetic product artifacts exercise the adapter and selector contract."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.protocol import (
    LOCAL_OPTIMIZER_POLICY,
    STRIP_PRODUCT_ORACLE_SCOPE,
    measurement_operator,
)
from benchmarks.database.adapters.linear import project
from benchmarks.database.model import digest
from benchmarks.dispatch.generate import _context
from tests.benchmark_database_fixtures import artifact
from tests.test_dispatch_generation import dataset_for


def strip_artifact(size=256):
    large = size > 1024
    run = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-{'large-' if large else ''}strip-{size}-rho3.json",
        f"benchmarks/cuda/linear/plans-profile-product-{'large-' if large else ''}strip.json",
    )
    run = replace(run, case=replace(run.case, rounds=3))
    snapshot = json.loads(json.dumps(run.snapshot()))
    old = artifact()
    meta = old["records"][0]["metadata"] | {
        "case": snapshot["case"],
        "input_hashes": snapshot["input_hashes"],
        "snapshot_sha256": digest(snapshot),
        "seed": run.case.seed,
        "polar_update": "torch",
    }
    result = old["records"][2]["result"] | {
        "operator": json.loads(
            json.dumps(asdict(measurement_operator(snapshot["case"])))
        ),
        "rows": run.case.rows,
        "atoms": run.case.atoms,
        "profile": run.case.profile,
        "optimizer": snapshot["case"]["optimizer"],
        "optimizer_policy": LOCAL_OPTIMIZER_POLICY,
    }
    records = []
    for kind in ("correctness", "measure"):
        for entry in snapshot["plans"]:
            records.append(
                {
                    "metadata": meta
                    | {"worker": kind, "plan_id": entry["id"], "plan": entry["plan"]},
                    "result": (
                        {
                            "status": "PASS",
                            "scope": STRIP_PRODUCT_ORACLE_SCOPE,
                            **{
                                k: {"max": 1e-6, "rel_l2": 1e-7}
                                for k in ("y", "dx", "dp", "polar_update")
                            },
                        }
                        if kind == "correctness"
                        else result | {"reference": entry["id"]}
                    ),
                }
            )
    records.append(
        {
            "metadata": meta | {"worker": "dense", "plan_id": None, "plan": None},
            "result": result
            | {
                "reference": "dense_linear",
                "atoms": None,
                "profile": None,
                "initial_p_sha256": None,
                "optimizer_policy": "ordinary AdamW",
            },
        }
    )
    return old | {
        "run": snapshot,
        "snapshot_sha256": digest(snapshot),
        "records": copy.deepcopy(records),
    }


def test_new_projection_and_generation_keep_product_contract():
    value = strip_artifact()
    projection = project(value)
    assert projection.revision == 5
    assert projection.protocol["revision"] == 5
    assert projection.protocol["correctness"] == STRIP_PRODUCT_ORACLE_SCOPE
    assert projection.protocol["polar_update"] == "torch"
    dataset, _ = dataset_for(value)
    row = dataset["records"][0]
    context = _context(row, "cuda_graph")
    assert context.operator.kernel.composition == "profile_product"
    assert context.parameter_dim == 4
    assert context.input_shape == (32, 256)
    assert context.operator.out_features == 64


def test_large_strip_width_export_keeps_series_hashes_times_and_peaks():
    import hashlib

    from benchmarks.cuda.linear.global_profile_product import compact_width_record
    from benchmarks.submissions.__main__ import validate_result
    from benchmarks.submissions.policy import DEFAULT_POLICY, load_policy

    value = strip_artifact(8192)
    value.update(
        execution_id="019c7714-3b77-74d1-9866-e1f484aae2ab",
        started_at="2026-10-07T16:00:00+00:00",
    )
    atoms = value["run"]["case"]["atoms"]
    initial = [3 + i / 999983 for i in range(atoms)]
    final = [s + 0.001 for s in initial]
    for record in value["records"]:
        if record["metadata"]["worker"] == "measure":
            record["result"]["sigma_updates"] = {
                "fixed": False,
                "source": "current polar activity every forward/replay",
                "initial": initial.copy(),
                "final": final.copy(),
                "changed_atoms": atoms,
                "max_abs_change": max(b - a for a, b in zip(initial, final)),
            }
    original = copy.deepcopy(value)
    policy = load_policy(DEFAULT_POLICY)
    raw = json.dumps(original, indent=2).encode()
    assert len(raw) > policy.max_file_bytes
    with pytest.raises(ValueError, match="oversized"):
        validate_result(raw, policy)
    value["records"] = [
        compact_width_record(r) if r["metadata"]["worker"] == "measure" else r
        for r in value["records"]
    ]
    validate_result(json.dumps(value, indent=2).encode(), policy)
    for before, after in zip(original["records"], value["records"], strict=True):
        assert before["metadata"] == after["metadata"]
        if before["metadata"]["worker"] != "measure":
            assert before == after
            continue
        assert before["result"]["sigma_updates"]["initial"] == initial
        assert before["result"]["sigma_updates"]["final"] == final
        update = after["result"]["sigma_updates"]
        assert update["atoms"] == atoms
        assert update["initial_range"] == [min(initial), max(initial)]
        assert update["final_range"] == [min(final), max(final)]
        assert (
            update["initial_sha256"]
            == hashlib.sha256(
                json.dumps(initial, separators=(",", ":")).encode()
            ).hexdigest()
        )
        assert (
            update["final_sha256"]
            == hashlib.sha256(
                json.dumps(final, separators=(",", ":")).encode()
            ).hexdigest()
        )
        assert {k: v for k, v in before["result"].items() if k != "sigma_updates"} == {
            k: v for k, v in after["result"].items() if k != "sigma_updates"
        }


@pytest.mark.parametrize("part", ["scope", "floor", "update"])
def test_rejects_mixed_legacy_oracle_and_normalization(part):
    value = strip_artifact()
    if part == "scope":
        value["records"][0]["result"]["scope"] = (
            "independent FP64 scalar Y/dX/all atom gradients; production polar update"
        )
    elif part == "floor":
        next(r for r in value["records"] if r["metadata"]["worker"] == "measure")[
            "result"
        ]["operator"]["kernel"]["normalization"]["floor"] = 0.5
    else:
        next(r for r in value["records"] if r["metadata"]["worker"] == "measure")[
            "result"
        ]["optimizer_policy"] = "ordinary AdamW"
    with pytest.raises(ValueError):
        project(value)
