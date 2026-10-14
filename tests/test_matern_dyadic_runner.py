"""Synthetic dispatch/assessment fixtures; fake bytes/timings are not GPU evidence."""

import copy
import json
import sys

import pytest

from benchmarks.cuda.linear import matern_dyadic_product as md
from benchmarks.cuda.linear import periodic_comparison as pc


def record(kind, *, verify=False):
    case = md.case_definition()
    gate = {
        "status": "PASS",
        "errors": {
            key: {"max_abs": 1e-6, "relative_l2": 1e-6} for key in ("Y", "dX", "all_dP")
        },
    }
    amount = {
        "matern-shared": 8000,
        "matern-stream1": 3000,
        "matern-stream2": 4000,
        "dense": 2000,
    }[kind]
    return {
        "schema": pc.worker_schema(case),
        "status": "PASS",
        "kind": kind,
        "size": 1024,
        "batch": 32,
        "atoms": None if kind == "dense" else 52428,
        "weight_shape": [1024, 1024],
        "case_definition": case,
        "actual_updates": 24,
        "runtime": {"gpu": "synthetic", "tf32": False, "dtype": "float32"},
        "timing": None if verify else {"median_ms": 1.0, "samples_ms": [1.0] * 21},
        "peak_capture_replay": {
            "allocated_bytes": amount,
            "reserved_bytes": amount * 2,
        },
        "initial_oracle": None if kind == "dense" else copy.deepcopy(gate),
        "updated_oracle": None if kind == "dense" else copy.deepcopy(gate),
        "width_changes": {"changed_atoms": 17},
        "user_sigma_changes": {"changed_atoms": 17},
        "verify_only": verify,
        "initial_hashes": {
            "x": "x",
            "dy": "dy",
            "target": "target",
            "parameters": "dense" if kind == "dense" else "atoms",
        },
        "source_hashes": {"synthetic.py": "synthetic"},
        "source_commit": "synthetic",
        "source_commit_hint": "synthetic",
        "case_sha256": "synthetic",
        "plan_catalog_sha256": "synthetic",
        "fixture": None
        if kind == "dense"
        else {
            "declaration_sha256": "same",
            "tensor_hashes": "same",
            "tensor_specs": "same",
        },
    }


def cohort():
    return [record(kind) for kind in pc.DYADIC_PRIMARY]


def test_memory_comparison_is_against_same_all_bank_math():
    result = pc.assess_fourier(cohort())
    assert result["status"] == "PASS"
    assert result["qualifies"] is False
    result = result["comparisons_against_all_bank"]
    assert result["matern-stream1"]["allocated_ratio"] == 0.375
    assert result["matern-stream2"]["allocated_reduction_bytes"] == 4000
    assert result["dense"]["allocated_ratio"] == 0.25


@pytest.mark.parametrize("kind", pc.DYADIC_PRIMARY)
def test_required_memory_controls_must_all_pass(kind):
    assert (
        pc.assess_fourier([r for r in cohort() if r["kind"] != kind])["status"]
        == "FAIL"
    )


def test_different_frozen_user_sigma_cohorts_refused():
    values = cohort()
    values[1]["case_definition"]["user_sigma_bounds"] = [0.01, 10]
    with pytest.raises(ValueError, match="frozen"):
        pc.assess_fourier(values)


def test_verify_only_has_no_memory_or_timing_adoption_claim():
    values = [record(kind, verify=True) for kind in pc.DYADIC_PRIMARY]
    assert pc.assess_fourier(values) == {
        "status": "PASS",
        "verification_only": True,
        "qualifies": False,
    }


def test_dyadic_cli_dispatch_without_cuda(tmp_path, monkeypatch):
    case = md.case_definition()
    path, output = tmp_path / "case.json", tmp_path / "worker.json"
    path.write_text(json.dumps(case))
    calls = []

    def fake_worker(actual, kind, *, phases, verify_only):
        calls.append((actual, kind, phases, verify_only))
        return record(kind)

    monkeypatch.setattr(pc, "fourier_worker", fake_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "periodic_comparison",
            "--case",
            str(path),
            "--worker",
            "matern-stream1",
            "--output",
            str(output),
        ],
    )
    pc.main()
    assert calls == [(case, "matern-stream1", False, False)]
    assert (
        json.loads(output.read_text())["schema"] == "matern-dyadic-worker.research.v1"
    )


def test_dyadic_plan_roundtrip_and_frozen_binding():
    from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog

    entries = decode_catalog(json.loads(pc.DYADIC_CATALOG.read_text()))
    assert {entry.id for entry in entries} == {
        "matern-shared",
        "matern-stream1",
        "matern-stream2",
    }
    model = md.fixture(md.case_definition(), atoms=7, size=128)
    for entry in entries:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan
        assert pc.bind_fourier_plan(
            model, entry.id, case=md.case_definition()
        ) == REGISTRY.dump_plan(entry.plan)
