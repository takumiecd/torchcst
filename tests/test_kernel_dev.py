"""Development tooling must reject invalid declarations and unavailable GPU gates."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.kernel_dev import __main__ as dev

LINEAR = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
PLANS = LINEAR / "plans-local-contraction.json"
CASE = LINEAR / "cases/local-contraction-64-sigma3.json"


def test_preflight_validates_multiple_cases_without_gpu_imports():
    code = f"""
import sys
from tools.kernel_dev.__main__ import main
assert main(['check', '--plans', {str(PLANS)!r},
    '--case', {str(CASE)!r}, '--case', {str(CASE)!r}]) == 0
assert 'triton' not in sys.modules
assert 'torchcst._backends.cuda.algorithms.local_product.executor' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    report = json.loads(result.stdout)
    assert report["scope"] == "declarations_only"
    assert len(report["cases"]) == 2
    assert report["cases"][0]["dense"] is True


def test_invalid_case_fails_instead_of_reporting_pass(tmp_path, capsys):
    value = json.loads(CASE.read_text())
    value["plans"] = ["missing"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    assert dev.main(["check", "--plans", str(PLANS), "--case", str(path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "unknown plan id" in captured.err


@pytest.mark.parametrize("suite", dev.GPU_SUITES)
def test_gpu_suite_rejects_cpu_host(suite, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        dev.subprocess, "run", lambda *a, **k: pytest.fail("ran pytest")
    )
    with pytest.raises(ValueError, match="requires NVIDIA CUDA and Triton"):
        dev.test_suite(suite)


def test_cpu_suite_preserves_pytest_failure_status(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=5)

    monkeypatch.setattr(dev.subprocess, "run", run)
    assert dev.test_suite("cpu") == 5
    assert calls == [
        (
            [sys.executable, "-m", "pytest", "-q", "tests"],
            {"cwd": dev.ROOT, "check": False},
        )
    ]


def test_gpu_suite_rejects_missing_triton(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="requires NVIDIA CUDA and Triton"):
        dev.test_suite("local-product")


def test_cpu_suite_rejects_visible_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        dev.test_suite("cpu")


@pytest.mark.parametrize(
    "plans,case,candidate",
    [
        (PLANS, CASE, "persistent-supportprep-band"),
        (
            LINEAR / "plans.json",
            LINEAR / "cases/normalized-1024-broad.json",
            "window512",
        ),
    ],
)
def test_prepared_comparison_runs_through_existing_manifest(
    tmp_path, plans, case, candidate
):
    output = tmp_path / "comparison"
    source = dev.load_run(case, plans)
    report = dev.prepare(plans, case, [candidate], output)
    run = dev.load_run(output / "case.json", output / "plans.json")
    assert run.case == source.case
    assert [entry.id for entry in run.plans] == [source.baseline, candidate]
    assert run.baseline == source.baseline and run.dense
    assert len(dev.decode_catalog(dev.read_json(output / "plans.json")[0])) == 2
    assert report["cases"][0]["input_hashes"] == run.input_hashes
    assert json.loads((output / "source.json").read_text())["input_hashes"] == (
        source.input_hashes
    )


@pytest.mark.parametrize(
    "candidates",
    [["unknown"], ["persistent-band", "persistent-band"], ["hybrid-persistent-mid4"]],
)
def test_prepare_rejects_invalid_candidates_before_writing(tmp_path, candidates):
    output = tmp_path / "comparison"
    with pytest.raises(ValueError):
        dev.prepare(PLANS, CASE, candidates, output)
    assert not output.exists()


def test_prepare_does_not_overwrite_existing_experiment(tmp_path):
    path = tmp_path / "case.json"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        dev.prepare(PLANS, CASE, ["persistent-band"], tmp_path)
    assert path.read_text() == "existing evidence"


def test_prepare_rejects_fixture_mismatch_before_writing(tmp_path):
    normalized = dev.read_json(LINEAR / "plans.json")[0]
    local = dev.read_json(PLANS)[0]
    catalog = tmp_path / "mixed.json"
    catalog.write_text(
        json.dumps(normalized | {"plans": normalized["plans"] + local["plans"]})
    )
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="mathematical contract differs"):
        dev.prepare(catalog, CASE, ["window512"], output)
    assert not output.exists()


def test_prepare_selects_new_catalog_candidate_and_requires_dense(tmp_path):
    value = dev.read_json(CASE)[0]
    value["plans"] = [value["baseline"]]
    case = tmp_path / "source-case.json"
    case.write_text(json.dumps(value))
    output = tmp_path / "comparison"
    report = dev.prepare(PLANS, case, ["persistent-band"], output)
    assert report["cases"][0]["plans"] == [value["baseline"], "persistent-band"]
    value["dense"] = False
    case.write_text(json.dumps(value))
    rejected = tmp_path / "without-dense"
    with pytest.raises(ValueError, match="dense reference"):
        dev.prepare(PLANS, case, ["persistent-band"], rejected)
    assert not rejected.exists()
