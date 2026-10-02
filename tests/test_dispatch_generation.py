"""Synthetic observations exercise ranking/export, never claim GPU performance."""

import copy
import json
import subprocess
import sys
import uuid
from dataclasses import replace

import pytest

from benchmarks.database.adapters.linear import ADAPTER, REVISION, project
from benchmarks.database.model import digest
from benchmarks.dispatch import ScorePolicy, generate
from benchmarks.dispatch.generate import _context
from benchmarks.dispatch.request import validate_request
from tests.benchmark_database_fixtures import artifact
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import (
    FULL,
    WINDOW,
)
from torchcst._backends.cuda.dispatch import load_selector, validate_selector_artifact


def measured(*, full=2.0, window=1.0, full_peak=1200, window_peak=1500):
    value = artifact() | {
        "execution_id": str(uuid.uuid4()),
        "started_at": "2026-10-02T00:00:00Z",
    }
    for worker in value["records"]:
        meta, result = worker["metadata"], worker["result"]
        if meta["worker"] == "measure":
            time = full if meta["plan_id"] == "full" else window
            for mode in ("eager", "graph"):
                result[mode] = {"median_ms": time, "samples_ms": [time] * 3}
            result["peak_allocated_capture_replay_bytes"] = (
                full_peak if meta["plan_id"] == "full" else window_peak
            )
            result["peak_reserved_capture_replay_bytes"] = max(
                2048, result["peak_allocated_capture_replay_bytes"]
            )
    return value


def request_for(row, **changes):
    return {
        "schema_version": 1,
        "revision": "test-v1",
        "dataset": {
            "adapter": ADAPTER,
            "adapter_revision": REVISION,
            "gpu": row["environment"]["gpu"],
            "source_id": row["source_id"],
        },
        "execution_mode": "cuda_graph",
        "min_runs": 1,
        "excluded_run_ids": [],
        "score_policy": {"id": "speed", "revision": "v1", "parameters": {}},
        "fallback_plan": REGISTRY.dump_plan(FULL),
    } | changes


def dataset_for(*values):
    rows = []
    for value in values:
        projection = project(value)
        run_id = digest(value)
        projection_id = digest(
            {"run_id": run_id, "adapter": ADAPTER, "revision": REVISION}
        )
        for ordinal, record in enumerate(projection.records):
            if record.kind != "measure":
                continue
            declaration = projection.plans[record.plan_alias]
            rows.append(
                {
                    "run_id": run_id,
                    "projection_id": projection_id,
                    "ordinal": ordinal,
                    "adapter": ADAPTER,
                    "adapter_revision": REVISION,
                    "case_id": projection.case_id,
                    "case_declaration": projection.case,
                    "protocol_id": digest(projection.protocol),
                    "protocol": projection.protocol,
                    "source_id": digest(record.source),
                    "source": record.source,
                    "environment_id": digest(record.environment),
                    "environment": record.environment,
                    "plan_id": digest(declaration),
                    "plan_declaration": declaration,
                    "kind": "measure",
                    "status": "PASS",
                    "run_status": "PASS",
                    "payload": record.payload,
                    "submission": None,
                    "metrics": [
                        {
                            "name": m.name,
                            "scope": m.scope,
                            "unit": m.unit,
                            "statistic": m.statistic,
                            "value": m.value,
                            "samples": m.samples,
                            "details": m.details,
                        }
                        for m in record.metrics
                    ],
                }
            )
    request = request_for(rows[0])
    return reseal(
        {"schema_version": 1, "selection": request["dataset"], "records": rows}
    ), request


def reseal(dataset):
    body = {k: dataset[k] for k in ("schema_version", "selection", "records")}
    return body | {"id": digest(body)}


def test_actual_selector_roundtrip_and_unobserved_fallback(monkeypatch):
    dataset, request = dataset_for(measured())
    result = generate(dataset, request, registry=REGISTRY)
    assert len(result.artifact["entries"]) == 1
    # Simulate the recorded target runtime; no CUDA kernel is executed here.
    from torchcst._backends.cuda.dispatch import exact

    monkeypatch.setattr(exact.torch, "__version__", "2.8.0")
    monkeypatch.setattr(exact, "version", lambda _: "3.4.0")
    selector = load_selector(json.dumps(result.artifact), registry=REGISTRY)
    context = _context(dataset["records"][0], "cuda_graph")
    decision = selector.select(context)
    assert decision.plan == WINDOW
    assert decision.evidence_ids == tuple(result.artifact["entries"][0]["evidence_ids"])
    assert (
        selector.select(replace(context, atom_count=context.atom_count + 1)).plan
        == FULL
    )


def test_cross_runtime_generation_does_not_weaken_runtime_load_checks():
    dataset, request = dataset_for(measured())
    result = generate(dataset, request, registry=REGISTRY)
    assert (
        validate_selector_artifact(result.artifact, registry=REGISTRY)
        == result.artifact
    )
    with pytest.raises(ValueError, match="runtime"):
        load_selector(json.dumps(result.artifact), registry=REGISTRY)


def test_eager_uses_eager_time_and_unknown_metric_can_be_scored():
    dataset, request = dataset_for(measured())
    request["execution_mode"] = "eager"
    for row in dataset["records"]:
        for metric in row["metrics"]:
            if metric["name"] == "step.time" and metric["scope"] == "eager":
                metric["value"] = (
                    1
                    if row["plan_declaration"]["algorithm_id"] == "normalized_full"
                    else 5
                )
        row["metrics"].append(
            {
                "name": "future.energy",
                "scope": "step",
                "unit": "joules",
                "statistic": "mean",
                "value": 1
                if row["plan_declaration"]["algorithm_id"] == "normalized_full"
                else 2,
                "samples": [],
                "details": {},
            }
        )
    dataset = reseal(dataset)
    result = generate(dataset, request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == digest(REGISTRY.dump_plan(FULL))
    assert result.artifact["entries"][0]["condition"]["execution_mode"] == "eager"
    assert all(
        m["summary"]["variance"] is None
        for m in result.leaderboard["leaderboards"][0]["candidates"][0]["metrics"]
    )
    request["score_policy"] = {"id": "energy", "revision": "v1", "parameters": {}}
    policy = ScorePolicy(
        "energy",
        "v1",
        {},
        lambda c: c.metric("future.energy", "step", "joules", "mean").mean,
    )
    assert generate(dataset, request, registry=REGISTRY, policy=policy).artifact[
        "entries"
    ][0]["plan_id"] == digest(REGISTRY.dump_plan(FULL))


def test_missing_metric_is_not_filled_or_ranked_from_partial_runs():
    dataset, request = dataset_for(
        measured(full=1, window=2), measured(full=1, window=2)
    )
    dataset["records"][0]["metrics"] = [
        m for m in dataset["records"][0]["metrics"] if m["name"] != "step.time"
    ]
    result = generate(reseal(dataset), request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == digest(
        REGISTRY.dump_plan(WINDOW)
    )


def test_multiple_runs_use_median_of_run_medians_and_report_dispersion():
    dataset, request = dataset_for(
        measured(full=1, window=3),
        measured(full=10, window=3),
        measured(full=10, window=3),
    )
    request["min_runs"] = 3
    result = generate(dataset, request, registry=REGISTRY)
    board = result.leaderboard["leaderboards"][0]
    assert board["winner_plan_id"] == digest(REGISTRY.dump_plan(WINDOW))
    assert [c["run_count"] for c in board["candidates"]] == [3, 3]
    full_time = next(
        m
        for m in board["candidates"][1]["metrics"]
        if m["name"] == "step.time" and m["scope"] == "graph"
    )
    assert full_time["summary"]["median"] == 10
    assert full_time["summary"]["stdev"] > 0


def test_memory_policy_and_speed_memory_ceiling():
    dataset, request = dataset_for(measured())
    request["score_policy"]["id"] = "memory"
    result = generate(dataset, request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == digest(REGISTRY.dump_plan(FULL))
    request["score_policy"] = {
        "id": "speed",
        "revision": "v1",
        "parameters": {"max_peak_allocated_bytes": 1300},
    }
    result = generate(dataset, request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == digest(REGISTRY.dump_plan(FULL))
    assert result.leaderboard["skipped"][0]["reason"] == "score function abstained"


def test_custom_function_gets_generic_metrics_and_cannot_mutate_evidence():
    dataset, request = dataset_for(measured())
    request["score_policy"] = {
        "id": "custom",
        "revision": "v7",
        "parameters": {"weight": 0.01},
    }
    original = copy.deepcopy(dataset)

    def score(candidate):
        time = candidate.metric("step.time", "graph", "ms", "median").median
        memory = candidate.metric(
            "memory.allocated", "capture_replay", "bytes", "peak"
        ).median
        candidate.observations[0]["submission"] = {"trust_points": 999}
        return time + memory * 0.01

    result = generate(
        dataset,
        request,
        registry=REGISTRY,
        policy=ScorePolicy("custom", "v7", {"weight": 0.01}, score),
    )
    assert result.artifact["entries"][0]["plan_id"] == digest(REGISTRY.dump_plan(FULL))
    assert dataset == original
    with pytest.raises(ValueError, match="custom functions"):
        generate(dataset, request, registry=REGISTRY)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "1"])
def test_invalid_custom_scores_rejected(score):
    dataset, request = dataset_for(measured())
    policy = ScorePolicy("speed", "v1", {}, lambda _: score)
    with pytest.raises(ValueError, match="finite number"):
        generate(dataset, request, registry=REGISTRY, policy=policy)


def test_snapshot_tampering_and_wrong_plan_hash_rejected():
    dataset, request = dataset_for(measured())
    dataset["records"][0]["plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot"):
        generate(dataset, request, registry=REGISTRY)
    with pytest.raises(ValueError, match="Plan identity"):
        generate(reseal(dataset), request, registry=REGISTRY)


@pytest.mark.parametrize(
    "field",
    [
        "case_id",
        "protocol_id",
        "source_id",
        "environment",
        "initial_inputs",
        "initial_p_sha256",
    ],
)
def test_incomparable_conditions_are_not_silently_combined(field):
    dataset, request = dataset_for(measured(), measured())
    row = dataset["records"][2]
    if field == "environment":
        row[field]["cuda"] = "different-cuda"
    elif field in ("initial_inputs", "initial_p_sha256"):
        row["payload"]["result"][field] = (
            {"x_sha256": "0" * 64, "target_sha256": "1" * 64}
            if field == "initial_inputs"
            else "0" * 64
        )
    else:
        row[field] = "0" * 64
    with pytest.raises(ValueError, match="cohorts|filters"):
        generate(reseal(dataset), request, registry=REGISTRY)


def test_failed_excluded_and_insufficient_observations_never_win():
    dataset, request = dataset_for(
        measured(full=1, window=2), measured(full=4, window=3)
    )
    request["excluded_run_ids"] = [dataset["records"][0]["run_id"]]
    result = generate(dataset, request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == digest(
        REGISTRY.dump_plan(WINDOW)
    )
    assert len(result.leaderboard["skipped"]) == 2
    request["min_runs"] = 2
    with pytest.raises(ValueError, match="no eligible"):
        generate(dataset, request, registry=REGISTRY)
    request["excluded_run_ids"] = []
    for row in dataset["records"][:2]:
        row["run_status"] = "FAIL"
    request["min_runs"] = 1
    assert generate(reseal(dataset), request, registry=REGISTRY).artifact["entries"][0][
        "plan_id"
    ] == digest(REGISTRY.dump_plan(WINDOW))


def test_duplicate_runs_and_local_device_indices():
    dataset, request = dataset_for(measured(), measured())
    dataset["records"].append(copy.deepcopy(dataset["records"][0]))
    for row in dataset["records"][2:4]:
        row["environment"]["device_index"] = 3
    result = generate(reseal(dataset), request, registry=REGISTRY)
    assert result.leaderboard["leaderboards"][0]["candidates"][0]["run_count"] == 2
    row = copy.deepcopy(dataset["records"][0])
    row["projection_id"] = "0" * 64
    dataset["records"].append(row)
    with pytest.raises(ValueError, match="multiple projections"):
        generate(reseal(dataset), request, registry=REGISTRY)


def test_order_independent_ranking_and_deterministic_ties():
    dataset, request = dataset_for(
        measured(full=1, window=1), measured(full=1, window=1)
    )
    result = generate(dataset, request, registry=REGISTRY)
    assert result.artifact["entries"][0]["plan_id"] == min(
        r["plan_id"] for r in dataset["records"]
    )
    dataset["records"].reverse()
    second = generate(reseal(dataset), request, registry=REGISTRY)
    assert second.artifact["entries"] == result.artifact["entries"]


@pytest.mark.parametrize(
    "change",
    [
        {"min_runs": True},
        {"execution_mode": "inference"},
        {"schema_version": 2},
        {"excluded_run_ids": ["bad"]},
    ],
)
def test_request_contract(change):
    _, request = dataset_for(measured())
    with pytest.raises(ValueError):
        validate_request(request | change)


def test_cli_offline_replay_and_no_overwrite(tmp_path):
    dataset, request = dataset_for(measured())
    request_path, dataset_path = tmp_path / "request.json", tmp_path / "dataset.json"
    request_path.write_text(json.dumps(request))
    dataset_path.write_text(json.dumps(dataset))
    command = [
        sys.executable,
        "-m",
        "benchmarks.dispatch",
        "--request",
        str(request_path),
        "--dataset",
        str(dataset_path),
        "--output",
        str(tmp_path / "generated"),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert {p.name for p in (tmp_path / "generated").iterdir()} == {
        "dispatch.json",
        "leaderboard.json",
        "dataset.json",
    }
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 1 and "already exists" in second.stderr
