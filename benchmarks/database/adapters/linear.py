"""Normalized v2 and local-product v3 projections of complete-step schema v1.

Validation checks internal consistency, not the submitter's authenticity or a
full-shape correctness certification. Failed runs retain evidence, no metrics.
"""

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot
from benchmarks.cuda.linear.protocol import (
    GLOBAL_PRODUCT_ORACLE_SCOPE,
    LOCAL_OPTIMIZER_POLICY,
    LOCAL_ORACLE_SCOPE,
    PRODUCT_ORACLE_SCOPE,
    SQUARE_STRIP_PRODUCT_ORACLE_SCOPE,
    STRIP_PRODUCT_ORACLE_SCOPE,
    measurement_operator,
)
from benchmarks.database.model import (
    Metric,
    Projection,
    Record,
    digest,
    execution_identity,
)

ADAPTER = "cuda.linear.complete-step"
REVISION = 2
LOCAL_REVISION = 3


def _hash(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected SHA256 hex string")


def _source(meta, *, research=False):
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
    label = None
    if commit != "unrecorded" and (
        type(commit) is not str or not re.fullmatch("[0-9a-f]{7,40}", commit)
    ):
        if (
            not research
            or type(commit) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", commit)
        ):
            raise ValueError("source commit needs a Git SHA/prefix or unrecorded")
        # Frozen research snapshots may have a working-tree label. Retain it
        # explicitly without certifying it as a Git commit; hashes identify code.
        label, commit = commit, "unrecorded"
    result = {
        "commit": commit,
        "commit_is_full_sha": commit != "unrecorded" and len(commit) == 40,
        "hashes": normalized,
    }
    if label is not None:
        result["recorded_source_label"] = label
    return result


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
    case = json.loads(json.dumps(asdict(run.case), allow_nan=False))
    strip = run.case.fixture in (
        "polar_profile_product_strip",
        "polar_profile_product_square_strip",
    )
    product = run.case.fixture in (
        "polar_profile_product",
        "polar_profile_product_strip",
        "polar_profile_product_square_strip",
        "polar_profile_product_global",
    )
    local = product or run.case.fixture == "local_polar_product"
    global_product = run.case.fixture == "polar_profile_product_global"
    square_strip = run.case.fixture == "polar_profile_product_square_strip"
    revision = (
        7
        if square_strip
        else 6
        if global_product
        else 5
        if strip
        else 4
        if product
        else LOCAL_REVISION
        if local
        else REVISION
    )
    oracle_scope = (
        SQUARE_STRIP_PRODUCT_ORACLE_SCOPE
        if square_strip
        else GLOBAL_PRODUCT_ORACLE_SCOPE
        if global_product
        else STRIP_PRODUCT_ORACLE_SCOPE
        if strip
        else PRODUCT_ORACLE_SCOPE
        if product
        else LOCAL_ORACLE_SCOPE
    )
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
    if local:
        protocol["correctness"] = oracle_scope
        protocol["optimizer_policy"] = LOCAL_OPTIMIZER_POLICY
    _hash(value["snapshot_sha256"])
    historical = (json.dumps(run.snapshot(), indent=2, allow_nan=False) + "\n").encode()
    # Older producer revisions used different dictionary order in Plan JSON.
    # Preserve that order from the original artifact to verify its byte hash.
    original = (json.dumps(value["run"], indent=2, allow_nan=False) + "\n").encode()
    if value["snapshot_sha256"] not in (
        digest(json.loads(historical)),
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
        protocol["initialization"] = {"method": "not verified"}
        return Projection(
            ADAPTER, revision, case, plans, run.baseline, protocol, tuple(records)
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
    op = measurement_operator(case)
    operator = json.loads(json.dumps(asdict(op)))
    common = None
    initialization = None
    inputs = None
    parameters = None
    polar_update = None
    validated = []
    for record in records:
        meta, result = record.payload["metadata"], record.payload["result"]
        actual_initialization = meta.get(
            "initialization", {"method": "torch.cuda.legacy-v1"}
        )
        if actual_initialization != {"method": "torch.cuda.legacy-v1"} and (
            type(actual_initialization) is not dict
            or set(actual_initialization) != {"method", "cpu_capability"}
            or actual_initialization["method"] != "torch.cpu.v1"
            or type(actual_initialization["cpu_capability"]) is not str
            or not actual_initialization["cpu_capability"]
        ):
            raise ValueError("unknown initialization method")
        if initialization is not None and initialization != actual_initialization:
            raise ValueError("workers use different initialization methods")
        initialization = actual_initialization
        protocol["initialization"] = initialization
        protocol["revision"] = 2 if initialization["method"] == "torch.cpu.v1" else 1
        if local:
            actual_update = meta.get("polar_update")
            if actual_update not in ("torch", "fused"):
                raise ValueError("unknown polar update implementation")
            if polar_update is not None and polar_update != actual_update:
                raise ValueError("workers use different polar update implementations")
            polar_update = actual_update
            protocol["polar_update"] = polar_update
            protocol["revision"] = (
                7
                if square_strip
                else 6
                if global_product
                else 5
                if strip
                else 4
                if product
                else 3
            )
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
        source, environment = _source(meta, research=local), _environment(meta)
        identity = (source, environment)
        if common is not None and identity != common:
            raise ValueError("workers have different source/runtime/GPU")
        common = identity
        if record.kind == "correctness":
            if not local and (
                result.get("fixture_sizes") != [1024, 4, 4]
                or result.get("fixture_atoms") != 8
                or result.get("scope") != protocol["correctness"]
            ):
                raise ValueError("unknown independent oracle fixture/scope")
            if local and result.get("scope") != oracle_scope:
                raise ValueError("unknown local independent oracle scope")
            metrics = tuple(
                Metric(
                    f"error.{tensor}",
                    "small_oracle",
                    "dimensionless",
                    key,
                    result[tensor][key],
                )
                for tensor in (
                    ("y", "dx", "dp", "polar_update")
                    if local
                    else ("w", "y", "dx", "dp")
                )
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
            if local and result.get("optimizer_policy") != (
                "ordinary AdamW" if dense else LOCAL_OPTIMIZER_POLICY
            ):
                raise ValueError("measurement polar optimizer contract differs")
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
        ADAPTER, revision, case, plans, run.baseline, protocol, tuple(validated)
    )
