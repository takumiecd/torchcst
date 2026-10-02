"""Projection v1 of the normalized CUDA complete-step artifact schema v1.

Validation checks internal consistency, not the submitter's authenticity or a
full-shape correctness certification. Failed runs retain evidence, no metrics.
"""

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict

from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.manifest import decode_snapshot
from benchmarks.database.model import (
    Metric,
    Projection,
    Record,
    digest,
    execution_identity,
)
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY

ADAPTER = "cuda.linear.complete-step"
REVISION = 1


def _hash(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected SHA256 hex string")


def _source(meta):
    hashes = meta.get("source_hashes")
    if type(hashes) is not dict or not hashes:
        raise ValueError("missing source hashes")
    normalized = {}
    for name, sha in hashes.items():
        _hash(sha)
        if type(name) is not str:
            raise ValueError("source paths must be strings")
        if "benchmarks/" in name:
            name = "benchmarks/" + name.rsplit("benchmarks/", 1)[1]
        elif "torchcst/" in name:
            name = "torchcst/" + name.rsplit("torchcst/", 1)[1]
        else:
            raise ValueError("unknown source path")
        if ".." in name.split("/") or name in normalized:
            raise ValueError("invalid/duplicate normalized source path")
        normalized[name] = sha
    if not any(name.startswith("torchcst/") for name in normalized):
        raise ValueError("missing package source hashes")
    commit = meta.get("source_commit")
    if commit != "unrecorded" and (
        type(commit) is not str or not re.fullmatch("[0-9a-f]{7,40}", commit)
    ):
        raise ValueError("source commit needs a Git SHA/prefix or unrecorded")
    return {
        "commit": commit,
        "commit_is_full_sha": commit != "unrecorded" and len(commit) == 40,
        "hashes": normalized,
    }


def _environment(meta):
    keys = (
        "gpu",
        "compute_capability",
        "sm_count",
        "total_device_bytes",
        "device_index",
        "torch",
        "cuda",
        "triton",
        "tf32",
        "dtype",
    )
    result = {key: meta[key] for key in keys}
    for name in ("gpu", "torch", "cuda", "triton"):
        if type(result[name]) is not str or not result[name]:
            raise ValueError("missing hardware/runtime metadata")
    for name in ("sm_count", "total_device_bytes", "device_index"):
        if type(result[name]) is not int or result[name] < (
            0 if name == "device_index" else 1
        ):
            raise ValueError("invalid GPU metadata")
    cc = result["compute_capability"]
    if (
        type(cc) is not list
        or len(cc) != 2
        or any(type(x) is not int or x < 0 for x in cc)
    ):
        raise ValueError("invalid compute capability")
    return result


def _metrics(result, rounds):
    metrics = []
    for mode in ("eager", "graph"):
        times = result[mode]
        samples = times["samples_ms"]
        if type(samples) is not list or len(samples) != rounds:
            raise ValueError("wrong time sample count")
        metric = Metric("step.time", mode, "ms", "median", times["median_ms"], samples)
        if any(x <= 0 for x in [metric.value, *samples]):
            raise ValueError("times must be positive")
        if not math.isclose(metric.value, statistics.median(samples), rel_tol=1e-12):
            raise ValueError("reported median differs from samples")
        metrics.append(metric)
    for field, name, statistic in (
        ("allocated_before_capture_bytes", "memory.allocated", "current"),
        ("peak_allocated_capture_replay_bytes", "memory.allocated", "peak"),
        ("peak_reserved_capture_replay_bytes", "memory.reserved", "peak"),
    ):
        number = result[field]
        if type(number) is not int or number < 0:
            raise ValueError("invalid memory measurement")
        scope = "before_capture" if statistic == "current" else "capture_replay"
        metrics.append(
            Metric(
                name,
                scope,
                "bytes",
                statistic,
                number,
                details={
                    "measurement_scope": result["memory_scope"],
                },
            )
        )
    if not (
        result["allocated_before_capture_bytes"]
        <= result["peak_allocated_capture_replay_bytes"]
        <= result["peak_reserved_capture_replay_bytes"]
    ):
        raise ValueError("inconsistent memory peaks")
    # This protocol never measures total process memory. Do not invent a value.
    if result.get("total_gpu_process_bytes") is not None:
        raise ValueError("process memory is unmeasured in this protocol")
    return tuple(metrics)


def project(value):
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("unsupported linear artifact schema")
    if (
        value.get("status") not in ("PASS", "FAIL")
        or type(value.get("records")) is not list
    ):
        raise ValueError("run must be completed with a record list")
    execution_identity(value)
    run = decode_snapshot(value["run"])
    case = asdict(run.case)
    plans = {p.id: REGISTRY.dump_plan(p.plan) for p in run.plans}
    protocol = {
        "id": ADAPTER,
        "revision": 1,
        "fixture": case["fixture"],
        "optimizer": case["optimizer"],
        "warmup": case["warmup"],
        "rounds": case["rounds"],
        "timing": "isolated-process synchronized wall time; sequential sessions",
        "correctness": "small mixed fixture; independent FP64 all-atom oracle",
        "memory": "warmed model, gradients and optimizer; capture/replay peak",
    }
    _hash(value["snapshot_sha256"])
    historical = (json.dumps(run.snapshot(), indent=2, allow_nan=False) + "\n").encode()
    # Older producer revisions used different dictionary order in Plan JSON.
    # Preserve that order from the original artifact to verify its byte hash.
    original = (json.dumps(value["run"], indent=2, allow_nan=False) + "\n").encode()
    if value["snapshot_sha256"] not in (
        digest(run.snapshot()),
        hashlib.sha256(historical).hexdigest(),
        hashlib.sha256(original).hexdigest(),
    ):
        raise ValueError("snapshot content/hash mismatch")
    records = []
    for record in value["records"]:
        if (
            type(record) is not dict
            or type(record.get("metadata")) is not dict
            or type(record.get("result")) is not dict
        ):
            raise ValueError("invalid worker record")
        meta, result = record["metadata"], record["result"]
        kind, alias = meta.get("worker"), meta.get("plan_id")
        if kind not in ("correctness", "measure", "dense") or result.get(
            "status"
        ) not in ("PASS", "FAIL"):
            raise ValueError("invalid worker kind/status")
        if (kind == "dense" and (alias is not None or not run.dense)) or (
            kind != "dense" and alias not in plans
        ):
            raise ValueError("unknown selected Plan/reference")
        records.append(Record(kind, result["status"], alias, {}, {}, record))
    if value["status"] == "FAIL":
        return Projection(
            ADAPTER, REVISION, case, plans, run.baseline, protocol, tuple(records)
        )

    performance = any(r.kind == "measure" for r in records)
    expected = {("correctness", p) for p in plans}
    if performance:
        expected |= {("measure", p) for p in plans}
        if run.dense:
            expected.add(("dense", None))
    actual = [(r.kind, r.plan_alias) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("incomplete/duplicate worker set")
    n = case["size"]
    h, j = (32, 32) if n == 1024 else (64, 128)
    op = operator_spec(
        sizes=(n, h, j),
        origin=(-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4),
        spacing=(1.0, 0.5, 0.5),
    )
    operator = json.loads(json.dumps(asdict(op)))
    common = None
    inputs = None
    parameters = None
    validated = []
    for record in records:
        meta, result = record.payload["metadata"], record.payload["result"]
        if (
            record.status != "PASS"
            or type(meta.get("schema_version")) is not int
            or meta["schema_version"] != 1
        ):
            raise ValueError("PASS run contains failed/invalid worker")
        if (
            meta.get("case") != case
            or meta.get("input_hashes") != run.input_hashes
            or meta.get("snapshot_sha256") != value["snapshot_sha256"]
        ):
            raise ValueError("worker differs from frozen run")
        if (
            meta.get("seed") != case["seed"]
            or meta.get("dtype") != "float32"
            or meta.get("tf32") is not False
        ):
            raise ValueError("worker precision/seed differs")
        if meta.get("plan") != plans.get(record.plan_alias):
            raise ValueError("worker Plan differs from selected Plan")
        source, environment = _source(meta), _environment(meta)
        identity = (source, environment)
        if common is not None and identity != common:
            raise ValueError("workers have different source/runtime/GPU")
        common = identity
        if record.kind == "correctness":
            if (
                result.get("fixture_sizes") != [1024, 4, 4]
                or result.get("fixture_atoms") != 8
                or result.get("scope") != protocol["correctness"]
            ):
                raise ValueError("unknown independent oracle fixture/scope")
            metrics = tuple(
                Metric(
                    f"error.{tensor}",
                    "small_oracle",
                    "dimensionless",
                    key,
                    result[tensor][key],
                )
                for tensor in ("w", "y", "dx", "dp")
                for key in ("max", "rel_l2")
            )
        else:
            if (
                result.get("operator") != operator
                or result.get("rows") != case["rows"]
                or result.get("optimizer") != case["optimizer"]
            ):
                raise ValueError("measurement operator/optimizer differs")
            dense = record.kind == "dense"
            if result.get("reference") != (
                "dense_linear" if dense else record.plan_alias
            ):
                raise ValueError("measurement reference differs")
            if result.get("atoms") != (None if dense else case["atoms"]) or result.get(
                "profile"
            ) != (None if dense else case["profile"]):
                raise ValueError("measurement initialization differs")
            initial = result["initial_inputs"]
            if type(initial) is not dict or set(initial) != {
                "x_sha256",
                "target_sha256",
            }:
                raise ValueError("missing input hashes")
            for sha in initial.values():
                _hash(sha)
            if inputs is not None and inputs != initial:
                raise ValueError("worker inputs differ")
            inputs = initial
            initial_p = result.get("initial_p_sha256")
            if dense:
                if initial_p is not None:
                    raise ValueError("dense reference has no atom parameters")
            else:
                _hash(initial_p)
                if parameters is not None and parameters != initial_p:
                    raise ValueError("candidate parameters differ")
                parameters = initial_p
            metrics = _metrics(result, case["rounds"])
        validated.append(
            Record(
                record.kind,
                record.status,
                record.plan_alias,
                environment,
                source,
                record.payload,
                metrics,
            )
        )
    return Projection(
        ADAPTER, REVISION, case, plans, run.baseline, protocol, tuple(validated)
    )
