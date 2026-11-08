"""A batch comparison must preserve Parameters and matching input/target rows."""

import hashlib
import inspect
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import frozen_replay as input_prefix
from benchmarks.cuda.linear import radius_cell_mass as radius
from benchmarks.cuda.linear import run
from benchmarks.cuda.linear.manifest import load_run


def test_frozen_replay_restores_p_and_original_report_but_records_regeneration(
    monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    case = load_run(
        root / "cases/radius-multirow-2048-r256-c8-xavier-q0.json",
        root / "plans-radius-multirow.json",
    ).case
    p = torch.linspace(0, 1, case.atoms * 4).reshape(-1, 4)
    x = torch.arange(256 * case.size, dtype=torch.float32).reshape(256, -1)
    report = {"scheme": "controlled-fixture"}
    model = SimpleNamespace(atoms=SimpleNamespace(p=p), initialization_report=report)
    payload = tmp_path / "inputs.pt"
    torch.save(
        {
            "p": p,
            "x": x,
            "target": -x,
            "dy": x + 1,
            "snapshot": {"case": asdict(case)},
            "initialization_report": report,
        },
        payload,
    )
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    calls = []

    def original(requested):
        calls.append(requested.rows)
        with torch.no_grad():
            p.add_(0.25)
        model.initialization_report = {"scheme": "different-current-host"}
        return model, None, None, None

    def metadata(*args):
        return {"existing": True}

    monkeypatch.setattr(radius, "fixture", original)
    monkeypatch.setattr(run, "_metadata", metadata)

    assert not inspect.signature(run.main).parameters
    original_argv = sys.argv

    def invoke():
        assert sys.argv[1:] == ["--forwarded"]
        chosen = replace(case, id="prefix-test", rows=64)
        actual, xx, yy, dy = radius.fixture(chosen)
        assert actual is model and actual.atoms.p is p
        torch.testing.assert_close(
            p, torch.linspace(0, 1, case.atoms * 4).reshape(-1, 4), atol=0, rtol=0
        )
        assert actual.initialization_report == report
        assert (
            torch.equal(xx, x[:64])
            and torch.equal(yy, -x[:64])
            and torch.equal(dy, (x + 1)[:64])
        )
        m = run._metadata(None, SimpleNamespace(case=chosen))
        assert m["existing"]
        replay = m["frozen_replay_policy"]
        assert replay["restored_parameters_sha256"] == radius.tensor_hash(p)
        assert replay["current_host_regeneration"][0][
            "regenerated_p_sha256"
        ] != radius.tensor_hash(p)
        assert replay["current_host_regeneration"][0][
            "regenerated_initialization_report"
        ] == {"scheme": "different-current-host"}
        assert (
            m["input_prefix_policy"]["selected_rows"] == 64
            and m["input_prefix_policy"]["input_payload_sha256"] == digest
        )

    monkeypatch.setattr(run, "main", invoke)
    input_prefix.main(
        [
            "--input-payload",
            str(payload),
            "--input-payload-sha256",
            digest,
            "--forwarded",
        ]
    )
    assert calls == [256] and radius.fixture is original and run._metadata is metadata
    assert sys.argv is original_argv
    with pytest.raises(ValueError, match="payload changed"):
        input_prefix.main(
            ["--input-payload", str(payload), "--input-payload-sha256", "0" * 64]
        )
