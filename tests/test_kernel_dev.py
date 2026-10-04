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
