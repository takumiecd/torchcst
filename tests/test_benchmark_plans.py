"""Plan-file contracts and subprocess orchestration without a GPU."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.cuda.linear import run as runner
from benchmarks.cuda.linear.manifest import (
    DEFAULT_PLANS,
    decode_catalog,
    load_run,
    load_snapshot,
    read_json,
)
from torchcst._backends.cuda.dispatch.select import FULL, WINDOW

CASE = DEFAULT_PLANS.parent / "cases/normalized-1024-broad.json"


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def test_catalog_resolves_exact_registered_recipes_and_snapshot(tmp_path):
    entries = decode_catalog(read_json(DEFAULT_PLANS)[0])
    assert [(e.id, e.plan) for e in entries] == [("full", FULL), ("window512", WINDOW)]
    run = load_run(CASE)
    snapshot = save(tmp_path / "snapshot.json", run.snapshot())
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert load_snapshot(snapshot, expected_hash=digest) == run
    snapshot.write_text(snapshot.read_text() + "\n")
    with pytest.raises(ValueError, match="hash differs"):
        load_snapshot(snapshot, expected_hash=digest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(schema_version=True),
        lambda c: c["plans"][0]["plan"].update(algorithm_id="missing"),
        lambda c: c["plans"][0]["plan"].update(algorithm_revision="v9"),
        lambda c: c["plans"][0]["plan"].update(schema_version=2),
        lambda c: c["plans"][0]["plan"]["recipe"].pop("support"),
        lambda c: c["plans"][0]["plan"]["recipe"].update(extra=True),
        lambda c: c["plans"][0]["plan"]["recipe"].update(atom_num_warps=True),
        lambda c: c["plans"][1]["plan"]["recipe"].update(window_rows=256),
        lambda c: c["plans"][1].update(id="full"),
        lambda c: c["plans"].append(
            {"id": "alias", "plan": copy.deepcopy(c["plans"][0]["plan"])}
        ),
        lambda c: c["plans"][0].update(id="../unsafe"),
    ],
)
def test_catalog_rejects_unknown_unvalidated_and_ambiguous_plans(mutate):
    value = read_json(DEFAULT_PLANS)[0]
    mutate(value)
    with pytest.raises((ValueError, TypeError)):
        decode_catalog(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(plans=["missing"]),
        lambda c: c.update(plans=["full", "full"]),
        lambda c: c.update(plans=["window512"]),
        lambda c: c.update(dense=1),
        lambda c: c["case"].update(rows=True),
        lambda c: c["case"].update(atoms=0),
        lambda c: c["case"].update(dtype="float16"),
        lambda c: c["case"].update(seed=-1),
        lambda c: c["case"].update(seed=2**63),
        lambda c: c["case"].update(warmup=0),
        lambda c: c["case"]["optimizer"].update(lr=float("inf")),
        lambda c: c["case"]["optimizer"].update(lr=True),
        lambda c: c["case"]["optimizer"].update(fused=False),
        lambda c: c["case"].update(unexpected=3),
    ],
)
def test_case_rejects_invalid_or_incomparable_conditions(tmp_path, mutate):
    value = read_json(CASE)[0]
    mutate(value)
    with pytest.raises((ValueError, TypeError)):
        load_run(save(tmp_path / "case.json", value))


@pytest.mark.parametrize(
    "raw", ['{"schema_version":1,"schema_version":1}', '{"value":NaN}']
)
def test_json_rejects_duplicate_keys_and_nonfinite_literals(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_json(path)


def test_cpu_cli_does_not_import_gpu_implementations():
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(CASE)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(k.endswith(('.executor', '.kernels', '.provider')) for k in sys.modules if 'normalized_euclidean_strip' in k)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@pytest.mark.parametrize(
    "failure", [None, "correctness", "mismatch", "missing", "malformed"]
)
def test_coordinator_freezes_case_and_stops_on_failure(tmp_path, monkeypatch, failure):
    case = save(tmp_path / "case.json", read_json(CASE)[0])
    output = tmp_path / "result.json"
    calls = []
    snapshots = []

    def worker(command, **kwargs):
        def flag(name):
            return command[command.index(name) + 1]

        kind = flag("--worker")
        plan_id = flag("--plan-id") if "--plan-id" in command else None
        calls.append((kind, plan_id))
        frozen = load_snapshot(
            flag("--snapshot"), expected_hash=flag("--snapshot-sha256")
        )
        snapshots.append(frozen)
        # Editing the source Case while a run is active must not affect workers.
        case.write_text('{"broken":true}')
        if failure == "missing":
            return SimpleNamespace(returncode=1)
        if failure == "malformed":
            Path(flag("--output")).write_text("{")
            return SimpleNamespace(returncode=0)
        result = {"status": "FAIL" if failure == kind else "PASS"}
        if kind != "correctness":
            result.update(
                initial_p_sha256="same",
                initial_inputs={"x": "same"},
                eager={"median_ms": 2.0},
                graph={"median_ms": 1.0},
            )
            if failure == "mismatch" and plan_id == "window512":
                result["initial_p_sha256"] = "different"
        save(
            Path(flag("--output")),
            {"metadata": {"worker": kind, "plan_id": plan_id}, "result": result},
        )
        return SimpleNamespace(returncode=1 if result["status"] == "FAIL" else 0)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    monkeypatch.setattr(
        sys, "argv", ["run", "--case", str(case), "--output", str(output)]
    )
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            runner.main()
    else:
        runner.main()
    combined = read_json(output)[0]
    assert combined["status"] == ("FAIL" if failure else "PASS")
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert [asdict(e.plan) for e in snapshots[0].plans] == [
        asdict(FULL),
        asdict(WINDOW),
    ]
    if failure in ("correctness", "missing", "malformed"):
        assert calls == [("correctness", "full")]
    elif failure == "mismatch":
        assert calls == [
            ("correctness", "full"),
            ("correctness", "window512"),
            ("measure", "full"),
            ("measure", "window512"),
        ]
    else:
        assert calls[-1] == ("dense", None)
        assert combined["comparisons"]["full"]["graph_time_ratio_to_baseline"] == 1.0
