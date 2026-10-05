"""Synthetic research artifacts; no stored GPU output or timing claims."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from benchmarks.cuda.linear.manifest import REGISTRY, load_run
from benchmarks.cuda.linear.protocol import (
    LOCAL_OPTIMIZER_POLICY,
    LOCAL_ORACLE_SCOPE,
    measurement_operator,
)
from benchmarks.database.adapters.linear import LOCAL_REVISION, project
from benchmarks.database.model import digest
from benchmarks.dispatch import generate
from benchmarks.dispatch.generate import _context
from tests.benchmark_database_fixtures import artifact
from tests.test_dispatch_generation import dataset_for, reseal
from torchcst._backends.dispatch import load_selector, validate_selector_artifact

LINEAR = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"


def local_artifact(profile="middle"):
    run = load_run(
        LINEAR / f"cases/local-contraction-64-{profile}.json",
        LINEAR / "plans-local-contraction.json",
    )
    run = replace(run, case=replace(run.case, rounds=3))
    snapshot = json.loads(json.dumps(run.snapshot()))
    value = artifact()
    meta = value["records"][0]["metadata"] | {
        "case": snapshot["case"],
        "input_hashes": snapshot["input_hashes"],
        "snapshot_sha256": digest(snapshot),
        "seed": run.case.seed,
        "polar_update": "fused",
        "source_commit": "de1a8c80-contraction-workingtree",
    }
    correctness = {
        "status": "PASS",
        "scope": LOCAL_ORACLE_SCOPE,
        **{k: {"max": 1e-6, "rel_l2": 1e-7} for k in ("y", "dx", "dp", "polar_update")},
    }
    measurement = value["records"][2]["result"] | {
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
    for entry in snapshot["plans"]:
        records.append(
            {
                "metadata": meta
                | {
                    "worker": "correctness",
                    "plan_id": entry["id"],
                    "plan": entry["plan"],
                },
                "result": copy.deepcopy(correctness),
            }
        )
    for entry in snapshot["plans"]:
        route = entry["plan"]["recipe"]["route"]
        time = (
            0.5
            if route == "persistent_supportprep_g"
            else (0.6 if route == "persistent_supportprep_band" else 1.0)
        )
        peak = (
            4000
            if route in ("persistent_saved_g", "persistent_supportprep_g")
            else 3000
        )
        records.append(
            {
                "metadata": meta
                | {"worker": "measure", "plan_id": entry["id"], "plan": entry["plan"]},
                "result": measurement
                | {
                    "reference": entry["id"],
                    "eager": {"median_ms": time, "samples_ms": [time] * 3},
                    "graph": {"median_ms": time, "samples_ms": [time] * 3},
                    "peak_allocated_capture_replay_bytes": peak,
                    "peak_reserved_capture_replay_bytes": 8192,
                },
            }
        )
    records.append(
        {
            "metadata": meta | {"worker": "dense", "plan_id": None, "plan": None},
            "result": measurement
            | {
                "reference": "dense_linear",
                "atoms": None,
                "profile": None,
                "initial_p_sha256": None,
                "optimizer_policy": "ordinary AdamW",
            },
        }
    )
    value.update(
        run=snapshot, snapshot_sha256=digest(snapshot), records=copy.deepcopy(records)
    )
    return value


def local_dataset(*values):
    dataset, request = dataset_for(*(values or (local_artifact(),)))
    request["fallback_plan"] = dataset["records"][0]["plan_declaration"]
    request["dataset"]["case_id"] = dataset["records"][0]["case_id"]
    dataset["selection"] = request["dataset"]
    return reseal(dataset), request


def test_local_projection_keeps_widths_updates_and_raw_diagnostics():
    value = local_artifact()
    p = project(value)
    assert p.revision == LOCAL_REVISION == 3
    assert len(p.plans) == 6 and len(p.records) == 13
    assert sum(len(r.metrics) for r in p.records) == 83
    assert p.case["widths"]["minimum"] < 1
    assert p.protocol["polar_update"] == "fused"
    assert p.records[0].source["commit"] == "unrecorded"
    assert not p.records[0].source["commit_is_full_sha"]
    assert p.records[0].source["recorded_source_label"].endswith("workingtree")
    assert {m.name for m in p.records[0].metrics} == {
        "error.y",
        "error.dx",
        "error.dp",
        "error.polar_update",
    }
    assert p.records[0].payload == value["records"][0]
    # Legacy normalized projections and identities remain revision 2.
    assert project(artifact()).revision == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v["records"][0]["metadata"].pop("polar_update"),
        lambda v: v["records"][0]["metadata"].update(polar_update="torch"),
        lambda v: v["records"][0]["result"].update(scope="full shape proven"),
        lambda v: v["records"][0]["result"].pop("polar_update"),
        lambda v: v["records"][0]["result"]["dp"].update(max=-1),
        lambda v: v["records"][6]["result"].update(optimizer_policy="ordinary AdamW"),
        lambda v: v["records"][-1]["result"].update(
            optimizer_policy=LOCAL_OPTIMIZER_POLICY
        ),
        lambda v: v["records"][6]["result"]["operator"]["kernel"]["parameterization"][
            "input_bounds"
        ].update(minimum=1),
        lambda v: v["records"][6]["metadata"].update(source_commit="not a source id"),
        lambda v: v["records"][6]["result"].update(total_gpu_process_bytes=9999),
    ],
)
def test_local_inconsistent_artifacts_are_rejected(mutate):
    value = local_artifact()
    mutate(value)
    with pytest.raises((ValueError, KeyError, TypeError)):
        project(value)


def test_failed_local_run_has_no_performance_and_correctness_only_is_valid():
    value = local_artifact()
    value["status"] = "FAIL"
    value["records"] = [
        {
            "metadata": {"worker": "measure", "plan_id": value["run"]["baseline"]},
            "result": {"status": "FAIL", "error": "compile failed"},
        }
    ]
    assert project(value).revision == 3
    assert not project(value).records[0].metrics
    value = local_artifact()
    value["records"] = value["records"][:6]
    assert len(project(value).records) == 6


def test_local_speed_and_baseline_peak_ceiling_select_different_plans(monkeypatch):
    dataset, request = local_dataset()
    speed = generate(dataset, request, registry=REGISTRY)
    request["score_policy"]["parameters"] = {"max_peak_allocated_bytes": 3000}
    bounded = generate(dataset, request, registry=REGISTRY)
    for result, route in (
        (speed, "persistent_supportprep_g"),
        (bounded, "persistent_supportprep_band"),
    ):
        entry = result.artifact["entries"][0]
        assert result.artifact["plans"][entry["plan_id"]]["recipe"]["route"] == route
        assert entry["condition"]["parameter_dim"] == 4
        assert (
            validate_selector_artifact(result.artifact, registry=REGISTRY)
            == result.artifact
        )
    from torchcst._backends.dispatch import exact

    monkeypatch.setattr(exact.torch, "__version__", "2.8.0")
    monkeypatch.setattr(exact, "version", lambda _: "3.4.0")
    selector = load_selector(json.dumps(bounded.artifact), registry=REGISTRY)
    context = _context(dataset["records"][0], "cuda_graph")
    assert selector.select(context).plan.recipe.route == "persistent_supportprep_band"
    assert (
        selector.select(replace(context, atom_count=17)).plan.recipe.route
        == "hybrid_persistent"
    )


def test_local_width_mixtures_are_separate_cohorts_not_hidden_runtime_labels():
    dataset, request = local_dataset(local_artifact(), local_artifact("narrow-only"))
    request["dataset"].pop("case_id")
    dataset["selection"] = request["dataset"]
    with pytest.raises(ValueError, match="multiple comparison cohorts"):
        generate(reseal(dataset), request, registry=REGISTRY)


def test_local_offline_cli_generates_with_research_registry(tmp_path):
    dataset, request = local_dataset()
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    (tmp_path / "request.json").write_text(json.dumps(request))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.dispatch",
            "--request",
            str(tmp_path / "request.json"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--output",
            str(tmp_path / "generated"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert (
        len(json.loads((tmp_path / "generated/dispatch.json").read_text())["entries"])
        == 1
    )
