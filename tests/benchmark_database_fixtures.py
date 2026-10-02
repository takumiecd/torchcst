"""Small complete artifact declarations; no CUDA execution or stored GPU output."""

import copy
import hashlib
import json
from dataclasses import asdict

from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.manifest import DEFAULT_PLANS, load_run


def artifact():
    run = load_run(DEFAULT_PLANS.parent / "cases/normalized-1024-broad.json")
    snapshot = run.snapshot()
    snapshot["case"]["rounds"] = 3
    sha = hashlib.sha256((json.dumps(snapshot, indent=2) + "\n").encode()).hexdigest()
    case = snapshot["case"]
    op = operator_spec(
        sizes=(1024, 32, 32), origin=(-511.5, -7.75, -7.75), spacing=(1.0, 0.5, 0.5)
    )
    meta = {
        "schema_version": 1,
        "case": case,
        "input_hashes": snapshot["input_hashes"],
        "snapshot_sha256": sha,
        "source_commit": "a" * 40,
        "source_hashes": {
            "src/torchcst/__init__.py": "b" * 64,
            "benchmarks/cuda/linear/run.py": "c" * 64,
        },
        "gpu": "Test GPU",
        "compute_capability": [8, 9],
        "sm_count": 58,
        "total_device_bytes": 24 * 1024**3,
        "device_index": 0,
        "torch": "2.8.0",
        "cuda": "12.8",
        "triton": "3.4.0",
        "tf32": False,
        "dtype": "float32",
        "seed": case["seed"],
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
                "result": {
                    "status": "PASS",
                    "scope": "small mixed fixture; independent FP64 all-atom oracle",
                    "fixture_sizes": [1024, 4, 4],
                    "fixture_atoms": 8,
                    **{
                        key: {"max": 1e-6, "rel_l2": 1e-7}
                        for key in ("w", "y", "dx", "dp")
                    },
                },
            }
        )
    for entry in [*snapshot["plans"], *([None] if snapshot["dense"] else [])]:
        alias = entry["id"] if entry else None
        records.append(
            {
                "metadata": meta
                | {
                    "worker": "measure" if entry else "dense",
                    "plan_id": alias,
                    "plan": entry["plan"] if entry else None,
                },
                "result": {
                    "status": "PASS",
                    "reference": alias or "dense_linear",
                    "operator": json.loads(json.dumps(asdict(op))),
                    "rows": case["rows"],
                    "atoms": case["atoms"] if entry else None,
                    "profile": case["profile"] if entry else None,
                    "initial_p_sha256": "d" * 64 if entry else None,
                    "initial_inputs": {"x_sha256": "e" * 64, "target_sha256": "f" * 64},
                    "optimizer": case["optimizer"],
                    "eager": {"median_ms": 2.0, "samples_ms": [1.0, 2.0, 3.0]},
                    "graph": {"median_ms": 0.5, "samples_ms": [0.4, 0.5, 0.6]},
                    "allocated_before_capture_bytes": 1000,
                    "peak_allocated_capture_replay_bytes": 1200,
                    "peak_reserved_capture_replay_bytes": 2048,
                    "total_gpu_process_bytes": None,
                    "memory_scope": "capture/replay includes warmed optimizer state",
                },
            }
        )
    return copy.deepcopy(
        {
            "schema_version": 1,
            "status": "PASS",
            "certification": "not assessed",
            "run": snapshot,
            "snapshot_sha256": sha,
            "records": [copy.deepcopy(r) for r in records],
        }
    )


def raw_artifact(value=None):
    return (
        json.dumps(artifact() if value is None else value, indent=2) + "\n"
    ).encode()
