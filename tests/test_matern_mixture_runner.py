"""CPU-only schema/dispatch checks using explicitly synthetic worker records.

These records exercise validation and comparison logic. Artificial timings,
memory and runtime labels are not measurements or evidence of GPU correctness.
"""

import copy
import hashlib
import json
import sys

import pytest

from benchmarks.cuda.linear import (
    fourier_profile_product as ff,
)
from benchmarks.cuda.linear import (
    matern_mixture_product as mm,
)
from benchmarks.cuda.linear import (
    periodic_comparison as pc,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_record(kind, *, verify_only=False):
    """Frozen v7 metadata, with explicit fake data for the validation unit test."""
    case = mm.case_definition()
    gate = {
        "status": "PASS",
        "errors": {
            key: {"max_abs": 1e-6, "relative_l2": 1e-6} for key in ("Y", "dX", "all_dP")
        },
    }
    milliseconds = {"matern-shared": 1.0, "matern-direct": 2.0, "dense": 0.5}[kind]
    allocated = {"matern-shared": 1000, "matern-direct": 2000, "dense": 500}[kind]
    return {
        "schema": "matern-mixture-worker.research.v7",
        "status": "PASS",
        "kind": kind,
        "size": 1024,
        "batch": 32,
        "atoms": None if kind == "dense" else 52428,
        "weight_shape": [1024, 1024],
        "case_definition": case,
        "actual_updates": 24,
        "schedule": case["schedule"],
        "optimizer": case["optimizer"],
        "runtime": {"gpu": "synthetic unit test", "tf32": False, "dtype": "float32"},
        "timing": None
        if verify_only
        else {
            "median_ms": milliseconds,
            "samples_ms": [milliseconds] * 21,
        },
        "peak_capture_replay": {
            "allocated_bytes": allocated,
            "reserved_bytes": allocated * 2,
        },
        "initial_oracle": None if kind == "dense" else copy.deepcopy(gate),
        "updated_oracle": None if kind == "dense" else copy.deepcopy(gate),
        "width_changes": {"changed_atoms": 17},
        "verify_only": verify_only,
        "initial_hashes": {
            "x": "synthetic x",
            "target": "synthetic target",
            "dy": "synthetic dy",
            "parameters": "synthetic dense parameters"
            if kind == "dense"
            else "synthetic CST parameters",
        },
        "source_hashes": {"src/synthetic.py": "synthetic source hash"},
        "source_commit": "synthetic source commit",
        "source_commit_hint": "synthetic source hint",
        "case_sha256": _sha(case),
        "plan_catalog_sha256": hashlib.sha256(
            pc.MATERN_CATALOG.read_bytes()
        ).hexdigest(),
        "fixture": None
        if kind == "dense"
        else {
            "declaration_sha256": "synthetic immutable declaration",
            "tensor_hashes": {"periods": "synthetic FP64 periods"},
            "tensor_specs": {"periods": {"shape": [2], "dtype": "torch.float64"}},
        },
    }


def synthetic_cohort(*, verify_only=False):
    return [
        synthetic_record(kind, verify_only=verify_only) for kind in pc.MATERN_PRIMARY
    ]


def test_profile_dispatch_and_legacy_worker_schemas_are_distinct():
    case = mm.case_definition()
    assert pc.profile_protocol(case) is mm
    assert pc.profile_catalog(case) == pc.MATERN_CATALOG
    assert pc.fourier_primary(case) == ("matern-shared", "matern-direct", "dense")
    assert pc.fourier_revision(case) == "v7"
    assert pc.worker_schema(case) == "matern-mixture-worker.research.v7"
    for legacy, revision in (
        (ff.case_definition(), "v4"),
        (ff.density_case_definition(), "v5"),
    ):
        assert pc.profile_protocol(legacy) is ff
        assert pc.profile_catalog(legacy) == pc.FOURIER_CATALOG
        assert pc.worker_schema(legacy) == f"fourier-worker.research.{revision}"


def test_matching_synthetic_cohort_assessment_keeps_dense_control():
    records = synthetic_cohort()
    for record in records:
        pc.validate_fourier_record(record)
    result = pc.assess_fourier(records)
    assert result["status"] == "PASS"
    assert result["qualifies"] is True  # Versus native direct, not dense.
    assert result["independent_runs"] == 1
    assert result["comparisons"]["matern-direct"]["time_ratio"] == 0.5
    assert result["comparisons"]["dense"]["time_ratio"] == 2.0
    assert set(result["comparisons"]) == {"matern-direct", "dense"}


def test_synthetic_verify_only_cannot_qualify_for_timing_selection():
    result = pc.assess_fourier(synthetic_cohort(verify_only=True))
    assert result == {"status": "PASS", "verification_only": True, "qualifies": False}


@pytest.mark.parametrize("verify_only", [False, True])
def test_timing_presence_must_match_verification_only_mode(verify_only):
    record = synthetic_record("matern-shared", verify_only=verify_only)
    record["verify_only"] = not verify_only
    with pytest.raises(ValueError, match="timing presence"):
        pc.validate_fourier_record(record)


def test_different_frozen_mixture_cases_cannot_share_comparison():
    records = synthetic_cohort()
    records[1]["case_definition"] = mm.case_definition(knots=8)
    with pytest.raises(ValueError, match="case definitions"):
        pc.assess_fourier(records)


def test_alternate_valid_recipe_cannot_replace_frozen_primary(tmp_path, monkeypatch):
    catalog = json.loads(pc.MATERN_CATALOG.read_text())
    catalog["plans"][0]["plan"]["recipe"]["block_sites"] = 128
    path = tmp_path / "altered-catalog.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setattr(pc, "MATERN_CATALOG", path)
    model = mm.fixture(mm.case_definition(), atoms=7, size=128)
    with pytest.raises(ValueError, match="frozen"):
        pc.bind_fourier_plan(model, "matern-shared")


@pytest.mark.parametrize("kind", ["matern-shared", "matern-direct", "dense"])
@pytest.mark.parametrize("status", ["FAIL", "OOM"])
def test_failures_cannot_publish_partial_timing(kind, status):
    record = {
        "kind": kind,
        "status": status,
        "failed_stage": "synthetic oracle failure",
        "error": "synthetic error",
        "timing": None,
    }
    pc.validate_fourier_record(record)
    record["timing"] = {"median_ms": 0.1}
    with pytest.raises(ValueError, match="without timings"):
        pc.validate_fourier_record(record)


def test_unknown_kind_refused_before_gpu_work():
    record = synthetic_record("matern-shared")
    record["kind"] = "unknown"
    with pytest.raises(ValueError, match="invalid"):
        pc.validate_fourier_record(record)
    with pytest.raises(ValueError, match="only"):
        pc.fourier_worker(mm.case_definition(), "unknown")


def test_old_fourier_worker_kind_refused_in_matern_cohort():
    with pytest.raises(ValueError, match="only"):
        pc.fourier_worker(mm.case_definition(), "fourier-shared")


@pytest.mark.parametrize("which", ["initial_oracle", "updated_oracle"])
def test_dropping_any_all_atom_gate_refused(which):
    record = synthetic_record("matern-shared")
    del record[which]["errors"]["all_dP"]
    with pytest.raises(ValueError, match="incomplete"):
        pc.validate_fourier_record(record)


@pytest.mark.parametrize("which", ["initial_oracle", "updated_oracle"])
@pytest.mark.parametrize("value", [4.0001e-4, float("nan"), float("inf"), -1e-6])
def test_fixed_numerical_gate_cannot_be_relaxed(which, value):
    record = synthetic_record("matern-shared")
    record[which]["errors"]["all_dP"]["max_abs"] = value
    with pytest.raises(ValueError, match="numerical"):
        pc.validate_fourier_record(record)


@pytest.mark.parametrize("key", ["x", "target", "dy", "parameters"])
def test_unmatched_shared_inputs_or_atom_parameters_refused(key):
    records = synthetic_cohort()
    records[1]["initial_hashes"][key] = "different synthetic tensor"
    with pytest.raises(ValueError, match="unmatched"):
        pc.assess_fourier(records)


@pytest.mark.parametrize(
    "key",
    [
        "source_hashes",
        "source_commit",
        "source_commit_hint",
        "case_sha256",
        "plan_catalog_sha256",
    ],
)
def test_unmatched_sources_or_frozen_case_hashes_refused(key):
    records = synthetic_cohort()
    records[1][key] = "different synthetic source"
    with pytest.raises(ValueError, match="unmatched"):
        pc.assess_fourier(records)


@pytest.mark.parametrize("key", ["declaration_sha256", "tensor_hashes", "tensor_specs"])
def test_unmatched_immutable_fixture_refused(key):
    records = synthetic_cohort()
    records[1]["fixture"][key] = "different synthetic fixture"
    with pytest.raises(ValueError, match="unmatched"):
        pc.assess_fourier(records)


@pytest.mark.parametrize("kind", pc.MATERN_PRIMARY)
def test_required_primary_control_cannot_be_dropped(kind):
    result = pc.assess_fourier([r for r in synthetic_cohort() if r["kind"] != kind])
    assert result["status"] == "FAIL"
    assert "missing" in result["reason"]


def test_cli_explicit_v7_worker_and_verify_only_dispatch_without_gpu(
    tmp_path, monkeypatch
):
    case = mm.case_definition()
    path, output = tmp_path / "case.json", tmp_path / "synthetic.json"
    path.write_text(json.dumps(case))
    calls = []

    def fake_worker(actual_case, kind, *, phases, verify_only):
        calls.append((actual_case, kind, phases, verify_only))
        return synthetic_record(kind, verify_only=verify_only)

    monkeypatch.setattr(pc, "fourier_worker", fake_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "periodic_comparison",
            "--case",
            str(path),
            "--worker",
            "matern-shared",
            "--verify-only",
            "--output",
            str(output),
        ],
    )
    pc.main()
    assert calls == [(case, "matern-shared", False, True)]
    result = json.loads(output.read_text())
    assert result["schema"] == "matern-mixture-worker.research.v7"
    assert result["timing"] is None


def test_cli_unknown_worker_refused_without_invoking_gpu(tmp_path, monkeypatch):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(mm.case_definition()))

    def never_called(*args, **kwargs):
        raise AssertionError("unknown worker must not reach execution")

    monkeypatch.setattr(pc, "fourier_worker", never_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "periodic_comparison",
            "--case",
            str(path),
            "--worker",
            "unknown",
            "--output",
            str(tmp_path / "output.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        pc.main()
    assert error.value.code == 2
