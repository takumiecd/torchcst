import copy
import hashlib
import json
from dataclasses import replace

import pytest

from benchmarks.cuda.linear.manifest import DEFAULT_PLANS, load_run
from benchmarks.cuda.linear.protocol import RATIONAL_WIDE_ORACLE_SCOPE
from benchmarks.database.adapters.linear import RATIONAL_WIDE_REVISION, project
from benchmarks.database.model import Metric, digest
from tests.benchmark_database_fixtures import artifact


def test_projection_separates_oracle_time_and_memory_scopes():
    p = project(artifact())
    assert len(p.plans) == 2
    assert len(p.records) == 5
    assert len(p.records[0].metrics) == 8
    measure = p.records[2]
    assert {(m.name, m.scope, m.unit, m.statistic) for m in measure.metrics} == {
        ("step.time", "eager", "ms", "median"),
        ("step.time", "graph", "ms", "median"),
        ("memory.allocated", "before_capture", "bytes", "current"),
        ("memory.allocated", "capture_replay", "bytes", "peak"),
        ("memory.reserved", "capture_replay", "bytes", "peak"),
    }
    assert all(m.scope != "eager" for m in measure.metrics if m.unit == "bytes")
    assert p.records[-1].kind == "dense" and p.records[-1].plan_alias is None
    assert p.records[-1].metrics


def test_rational_projection_keeps_three_plans_and_the_dense_control():
    root = DEFAULT_PLANS.parent
    run = load_run(
        root / "cases/rational-wide-4096-r128-c32-dispatch.json",
        root / "plans-rational-wide.json",
    )
    snapshot = run.snapshot()
    snapshot["case"]["rounds"] = 3
    snapshot_sha = hashlib.sha256((json.dumps(snapshot, indent=2) + "\n").encode()).hexdigest()
    data = artifact()
    meta = data["records"][0]["metadata"] | {
        "case": snapshot["case"],
        "input_hashes": snapshot["input_hashes"],
        "snapshot_sha256": snapshot_sha,
        "seed": run.case.seed,
        "polar_update": "torch",
    }
    measure = data["records"][2]["result"] | {
        "initial_inputs": data["records"][2]["result"]["initial_inputs"] | {"dy_sha256": "1" * 64},
        "optimizer": snapshot["case"]["optimizer"],
        "graph": None,
        "peak_allocated_training_bytes": 1024,
        "peak_reserved_training_bytes": 2048,
        "optimizer_implementation": "torchcst.optim.adamw.CSTAdamW",
    }
    oracle = {"status": "PASS", "scope": RATIONAL_WIDE_ORACLE_SCOPE, "errors": {
        name: {"max_abs": 1e-6, "relative_l2": 1e-7} for name in ("Y", "dX", "all_dP")
    }}
    records = []
    for entry in snapshot["plans"]:
        selected = meta | {"plan_id": entry["id"], "plan": entry["plan"]}
        records += [
            {"metadata": selected | {"worker": "correctness"}, "result": oracle},
            {"metadata": selected | {"worker": "measure"}, "result": measure | {"reference": entry["id"]}},
        ]
    dense = measure | {
        "reference": "dense_linear",
        "optimizer_implementation": "torch.optim.adamw.AdamW",
        "initial_effective_dense_weight_sha256": "2" * 64,
    }
    records.append({
        "metadata": meta | {"worker": "dense", "plan_id": None, "plan": None},
        "result": dense,
    })
    data.update(run=snapshot, snapshot_sha256=snapshot_sha, records=records)
    projection = project(data)
    assert projection.revision == RATIONAL_WIDE_REVISION
    assert len(projection.plans) == 3
    assert projection.records[-1].plan_alias is None
    assert {metric.scope for metric in projection.records[-1].metrics} == {"eager"}

    dense["initial_p_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="candidate parameters differ"):
        project(data)


def test_initialization_protocol_distinguishes_legacy_gpu_and_cpu_batches():
    data = artifact()
    legacy = project(data)
    assert legacy.protocol["initialization"] == {"method": "torch.cuda.legacy-v1"}
    for record in data["records"]:
        record["metadata"]["initialization"] = {
            "method": "torch.cpu.v1",
            "cpu_capability": "NO AVX",
        }
    cpu = project(data)
    assert cpu.protocol["revision"] == 2
    assert digest(cpu.protocol) != digest(legacy.protocol)
    data["records"][0]["metadata"].pop("initialization")
    with pytest.raises(ValueError, match="different initialization"):
        project(data)
    data["records"][0]["metadata"]["initialization"] = {"method": "unknown"}
    with pytest.raises(ValueError, match="unknown initialization"):
        project(data)


def test_failed_run_and_correctness_only_are_retained_without_performance():
    data = artifact()
    data["status"] = "FAIL"
    data["records"] = [
        {
            "metadata": {"worker": "measure", "plan_id": "full"},
            "result": {"status": "FAIL", "error": "out of memory"},
        }
    ]
    p = project(data)
    assert p.records[0].payload == data["records"][0]
    assert not p.records[0].metrics
    data = artifact()
    data["records"] = data["records"][:2]
    assert len(project(data).records) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(status="RUNNING"),
        lambda v: v.update(schema_version=True),
        lambda v: v.update(
            execution_id="not-a-uuid", started_at="2026-10-02T00:00:00Z"
        ),
        lambda v: v.update(execution_id="12345678-1234-1234-1234-123456789012"),
        lambda v: v.update(
            execution_id="12345678-1234-1234-1234-123456789012",
            started_at="2026-10-02T00:00:00",
        ),
        lambda v: v.update(snapshot_sha256="0" * 64),
        lambda v: v["records"].pop(0),
        lambda v: v["records"].append(copy.deepcopy(v["records"][0])),
        lambda v: v["records"][0]["result"].update(status="FAIL"),
        lambda v: v["records"][0]["metadata"].update(tf32=True),
        lambda v: v["records"][0]["metadata"]["plan"].update(algorithm_revision="v2"),
        lambda v: v["records"][0]["metadata"].update(source_commit="some-branch"),
        lambda v: v["records"][0]["metadata"]["source_hashes"].update(
            {"src/torchcst/__init__.py": "0" * 64}
        ),
        lambda v: v["records"][0]["metadata"].update(cuda="12.9"),
        lambda v: v["records"][0]["metadata"].update(total_device_bytes=1),
        lambda v: v["records"][0]["result"].update(fixture_atoms=128),
        lambda v: v["records"][0]["result"]["dp"].update(max=-1),
        lambda v: v["records"][2]["result"]["eager"].update(median_ms=20),
        lambda v: v["records"][2]["result"]["eager"]["samples_ms"].pop(),
        lambda v: v["records"][2]["result"]["graph"]["samples_ms"].__setitem__(0, 0),
        lambda v: v["records"][2]["result"].update(
            peak_reserved_capture_replay_bytes=100
        ),
        lambda v: v["records"][2]["result"].update(
            peak_allocated_capture_replay_bytes=True
        ),
        lambda v: v["records"][2]["result"].update(initial_p_sha256="0" * 64),
        lambda v: v["records"][-1]["result"]["initial_inputs"].update(
            x_sha256="0" * 64
        ),
        lambda v: v["records"][2]["result"].update(total_gpu_process_bytes=4000),
        lambda v: v["records"][2]["result"].update(atoms=1),
        lambda v: v["records"][2]["result"]["optimizer"].update(lr=0.2),
    ],
)
def test_inconsistent_pass_artifacts_are_rejected(mutate):
    data = artifact()
    mutate(data)
    with pytest.raises((ValueError, KeyError, TypeError)):
        project(data)


def test_cases_ignore_display_names_and_short_commits_are_explicitly_unresolved():
    p = project(artifact())
    assert replace(p, case=p.case | {"id": "renamed"}).case_id == p.case_id
    data = artifact()
    for r in data["records"]:
        r["metadata"]["source_commit"] = "abcdef0"
    p = project(data)
    assert p.records[0].source["commit_is_full_sha"] is False


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_extensible_metrics_reject_invalid_numbers(value):
    with pytest.raises(ValueError):
        Metric("forward.time", "kernel", "us", "median", value)


def test_new_metric_needs_no_fixed_time_or_memory_fields():
    m = Metric(
        "forward.time", "kernel", "us", "p95", 10, [8, 10], {"clock": "cuda_event"}
    )
    assert m.statistic == "p95" and m.details["clock"] == "cuda_event"
