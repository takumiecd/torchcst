"""Plan-file contracts and subprocess orchestration without a GPU."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from benchmarks.cuda.linear import run as runner
from benchmarks.cuda.linear.manifest import (
    DEFAULT_PLANS,
    decode_catalog,
    load_run,
    load_snapshot,
    read_json,
)
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import (
    FULL,
    WINDOW,
)

CASE = DEFAULT_PLANS.parent / "cases/normalized-1024-broad.json"


def test_cpu_batch_initialization_is_independent_of_global_rng():
    first = runner.generate_inputs(21, 4, 16)
    torch.manual_seed(999)
    torch.randn(20)
    second = runner.generate_inputs(21, 4, 16)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in first)
    assert not torch.equal(*first)
    assert not torch.equal(first[0], runner.generate_inputs(22, 4, 16)[0])


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
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.plans import FULL, WINDOW
for plan in (FULL, WINDOW):
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
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
    assert str(UUID(combined["execution_id"])) == combined["execution_id"]
    assert datetime.fromisoformat(combined["started_at"]).utcoffset() == timedelta(0)
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


LOCAL_PLANS = DEFAULT_PLANS.with_name("plans-local-product.json")


def test_local_catalog_roundtrip_and_fixture_contract(tmp_path):
    local_case = DEFAULT_PLANS.parent / "cases/local-32-mixed.json"
    run = load_run(local_case, LOCAL_PLANS)
    assert run.case.atoms == 51
    assert run.baseline == "local-torch"
    snapshot = save(tmp_path / "snapshot.json", run.snapshot())
    assert load_snapshot(snapshot) == run
    with pytest.raises(ValueError, match="contract differs"):
        value = read_json(CASE)[0]
        value.update(plans=["local-fused"], baseline="local-fused")
        load_run(save(tmp_path / "bad.json", value), LOCAL_PLANS)
    with pytest.raises(ValueError, match="contract differs"):
        value = read_json(local_case)[0]
        value.update(plans=["full"], baseline="full")
        load_run(save(tmp_path / "bad2.json", value))


def test_local_cpu_cli_does_not_import_triton():
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(DEFAULT_PLANS.parent / "cases/local-16-sharp.json")!r},
            '--plans', {str(LOCAL_PLANS)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert 'torchcst._backends.cuda.algorithms.local_product.kernels' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@pytest.mark.parametrize("size", [16, 32, 64, 128])
@pytest.mark.parametrize("profile", ["sharp", "few", "broad", "wide", "mixed"])
def test_local_initialization_covers_all_support_scales(size, profile):
    from benchmarks.cuda.linear.fixtures import local_product_state
    from benchmarks.cuda.linear.local_product import initialize
    from torchcst._backends.cuda.algorithms.local_product.preparation import decode

    run = load_run(
        DEFAULT_PLANS.parent
        / f"cases/local-{size}-{profile}{'-hybrid' if size == 128 else ''}.json",
        DEFAULT_PLANS.with_name("plans-local-hybrid.json")
        if size == 128
        else LOCAL_PLANS,
    )
    p = initialize(run.case)
    assert p.shape == (round(0.05 * size * size), 4)
    assert torch.equal(p, initialize(run.case))
    q = decode(local_product_state(birth=1), p)
    sigma = q[:, 1].rsqrt()
    assert ((sigma >= 1 - 1e-6) & (sigma <= 16 + 1e-5)).all()
    if profile != "mixed":
        target = {"sharp": 1, "few": 2, "broad": 3, "wide": 16}[profile]
        torch.testing.assert_close(
            sigma, torch.full_like(sigma, target), atol=2e-5, rtol=2e-6
        )

    if profile == "sharp":
        from torchcst._backends.cuda.algorithms.local_product.contract import Domain
        from torchcst._backends.cuda.algorithms.local_product.support import summarize

        report = summarize(q, Domain(size, size))
        assert report["onehot_both_live_atoms"] == len(p)


@pytest.mark.parametrize("size", [64, 128])
def test_rho_sweep_only_changes_initial_radius(size, tmp_path):
    from benchmarks.cuda.linear.fixtures import local_product_state
    from benchmarks.cuda.linear.local_product import initialize
    from torchcst._backends.cuda.algorithms.local_product.preparation import decode

    reference = None
    for suffix, rho in [
        ("1", 1),
        ("1_5", 1.5),
        ("2", 2),
        ("3", 3),
        ("4", 4),
        ("8", 8),
        ("16", 16),
    ]:
        run = load_run(
            DEFAULT_PLANS.parent / f"cases/local-{size}-rho{suffix}.json",
            DEFAULT_PLANS.with_name("plans-local-rho.json"),
        )
        p = initialize(run.case)
        q = decode(local_product_state(birth=1), p)
        torch.testing.assert_close(
            q[:, 1].rsqrt(), torch.full_like(q[:, 1], rho), atol=2e-5, rtol=2e-6
        )
        if reference is None:
            reference = p, q
        else:
            torch.testing.assert_close(p[:, 2:], reference[0][:, 2:], rtol=0, atol=0)
            torch.testing.assert_close(
                q[:, 0], reference[1][:, 0], rtol=2e-6, atol=2e-6
            )
        assert {entry.plan.recipe.route for entry in run.plans} >= {
            "polar_support",
            "polar_support_saved",
        }
        snapshot = save(tmp_path / f"rho{suffix}.json", run.snapshot())
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert load_snapshot(snapshot, expected_hash=digest) == run
