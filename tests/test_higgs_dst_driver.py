"""Fail-closed pilot qualification and snapshot/data provenance on CPU."""

import hashlib
import json
import subprocess
from argparse import Namespace

import pytest

from benchmarks.training import higgs_dst_driver as driver


def test_snapshot_verifies_data_and_the_executed_driver(tmp_path):
    source = tmp_path / "source"
    inputs = source / "research-inputs"
    inputs.mkdir(parents=True)
    code = source / "benchmarks/training/higgs_dst_driver.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"frozen driver")
    data = inputs / "screen.npz"
    data.write_bytes(b"normalized screen dataset")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    commit = "a" * 40
    (inputs / "source-manifest.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "files": {code.relative_to(source).as_posix(): sha(code)},
            }
        )
    )
    (inputs / "data-manifest.json").write_text(
        json.dumps({"artifacts": {"screen.npz": {"sha256": sha(data)}}})
    )
    executed = tmp_path / "__pool_driver__.py"
    executed.write_bytes(code.read_bytes())
    binding = driver.verify_inputs(source, commit, executed)
    assert binding["dataset_sha256"] == sha(data)
    executed.write_bytes(b"modified executed script")
    with pytest.raises(ValueError, match="executed pool driver"):
        driver.verify_inputs(source, commit, executed)
    executed.write_bytes(code.read_bytes())
    data.write_bytes(b"different dataset")
    with pytest.raises(ValueError, match="screen archive"):
        driver.verify_inputs(source, commit, executed)


def test_failed_qualification_prevents_training_and_preserves_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    monkeypatch.setenv("CST_JOB_OUTPUT", str(output))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        driver, "verify_inputs", lambda *args: {"source_commit": "a" * 40}
    )
    monkeypatch.setattr(driver, "probe_runtime", lambda env: {"gpu": "NVIDIA L4"})
    calls = []

    def fail_qualification(command, **kwargs):
        calls.append(command)
        assert "--qualify-only" in command
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(driver.subprocess, "run", fail_qualification)
    with pytest.raises(subprocess.CalledProcessError):
        driver.run(Namespace(source_commit="a" * 40, stage="pilot"))
    saved = json.loads((output / "dst-pilot-driver.json").read_text())
    assert saved["status"] == "failed"
    assert saved["confirmation_evaluated"] is False
    assert saved["test_evaluated"] is False
    assert len(calls) == len(saved["commands"]) == 1
