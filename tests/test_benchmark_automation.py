"""Request/source binding, result admission, and shared-pool adapter boundaries."""

import copy
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.automation import colab
from benchmarks.automation.contract import (
    decode,
    digest,
    encode,
    file_hash,
    prepare,
    validate_request,
    verify_source,
)
from benchmarks.automation.ingest import expected_worker_source, validate_bundle
from tools.colab.profiles import validate_hardware
from tests.benchmark_database_fixtures import artifact

REPO = Path(__file__).resolve().parents[1]


def make_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in (
        "plans.json",
        "cases/normalized-1024-broad.json",
        "cases/normalized-1024-sharp.json",
    ):
        path = root / "benchmarks/cuda/linear" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = decode((REPO / "benchmarks/cuda/linear" / name).read_bytes())
        if "case" in value:
            value["case"]["rounds"] = 3
        path.write_bytes(encode(value))
    for name in ("src/torchcst/__init__.py", "benchmarks/cuda/linear/run.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source identity fixture\n")
    (root / ".gitignore").write_text("output/\n")
    config = decode((REPO / "benchmarks/requests/colab-linear.json").read_bytes())
    (root / "request.json").write_bytes(encode(config))
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    return root


@pytest.fixture
def source(tmp_path):
    return make_source(tmp_path)


def prepared(root, *, cases=None, targets=None, repetitions=1):
    config = decode((root / "request.json").read_bytes())
    if cases is not None:
        config["cases"] = [
            "benchmarks/cuda/linear/cases/" + case + ".json" for case in cases
        ]
    if targets is not None:
        config["targets"] = targets
    config["repetitions"] = repetitions
    (root / "request.json").write_bytes(encode(config))
    subprocess.run(["git", "-C", str(root), "add", "request.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "request",
        ],
        check=True,
    )
    return prepare(root, "request.json")


def make_bundle(root, request, *, failure=False):
    root.mkdir(parents=True)
    request_file = root / "request.json"
    request_file.write_bytes(encode(request))
    results = root / "results"
    (results / "records").mkdir(parents=True)
    entries = []
    for job in request["request"]["jobs"]:
        value = artifact()
        value["execution_id"] = str(uuid.uuid4())
        value["started_at"] = "2026-10-02T00:00:00+00:00"
        value["run"] = copy.deepcopy(job["run"])
        value["snapshot_sha256"] = hashlib.sha256(
            (json.dumps(value["run"], indent=2) + "\n").encode()
        ).hexdigest()
        for record in value["records"]:
            record["metadata"].update(request["request"]["environment"])
            hardware = {
                "T4": ("Tesla T4", [7, 5], 40),
                "L4": ("NVIDIA L4", [8, 9], 58),
                "A100": ("NVIDIA A100-SXM4-40GB", [8, 0], 108),
                "H100": ("NVIDIA H100 80GB HBM3", [9, 0], 132),
                "G4": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", [12, 0], 148),
            }[job["accelerator"]]
            record["metadata"].update(
                {
                    "gpu": hardware[0],
                    "compute_capability": hardware[1],
                    "sm_count": hardware[2],
                }
            )
            record["metadata"].update(
                {
                    "source_commit": request["request"]["source_commit"],
                    "source_hashes": expected_worker_source(request),
                    "case": value["run"]["case"],
                    "input_hashes": value["run"]["input_hashes"],
                    "snapshot_sha256": value["snapshot_sha256"],
                }
            )
        if failure:
            value["status"] = "FAIL"
            value["records"] = [
                {
                    "metadata": {"worker": "correctness", "plan_id": "full"},
                    "result": {"status": "FAIL", "error": "GPU unavailable"},
                }
            ]
        path = results / f"records/{job['key']}.json"
        path.write_text(json.dumps(value, indent=2) + "\n")
        entries.append(
            {
                "key": job["key"],
                "pool_job_id": "l4job-" + uuid.uuid4().hex,
                "status": "failed" if failure else "succeeded",
                "source_archive_sha256": "a" * 64,
                "result_archive_sha256": "b" * 64,
                "artifact": f"records/{job['key']}.json",
                "artifact_sha256": file_hash(path),
            }
        )
    (results / "submission.json").write_bytes(
        encode(
            {"schema_version": 1, "request_id": request["request_id"], "jobs": entries}
        )
    )
    return request_file, results


def rewrite_result(results, edit, index=0, *, refresh_hash=True):
    bundle = decode((results / "submission.json").read_bytes())
    row = bundle["jobs"][index]
    path = results / row["artifact"]
    value = decode(path.read_bytes())
    edit(value)
    path.write_text(json.dumps(value, indent=2) + "\n")
    if refresh_hash:
        row["artifact_sha256"] = file_hash(path)
    (results / "submission.json").write_bytes(encode(bundle))


def test_freezes_candidates_source_and_independent_repetitions(source):
    request = prepared(
        source, cases=["normalized-1024-broad", "normalized-1024-sharp"], repetitions=3
    )
    assert len(request["request"]["jobs"]) == 6
    assert len(request["request"]["source_commit"]) == 40
    assert request == prepare(source, "request.json")
    first = request["request"]["jobs"][0]["run"]
    assert [p["id"] for p in first["plans"]] == ["full", "window512"]
    assert first["baseline"] == "full" and first["dense"]
    verify_source(source, request, checkout=True)
    (source / "src/torchcst/__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="commit source"):
        prepare(source, "request.json")
    with pytest.raises(ValueError):
        verify_source(source, request, checkout=True)


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r["request"].update(timeout_seconds=0),
        lambda r: r["request"].update(environment={"torch": "latest"}),
        lambda r: r["request"]["targets"][0].update(accelerator="RTX4090"),
        lambda r: r["request"]["source_files"].update({"../outside": "a" * 64}),
        lambda r: r["request"]["jobs"].append(r["request"]["jobs"][0]),
        lambda r: r["request"]["jobs"][0].update(repetition=True),
    ],
)
def test_rejects_unsupported_or_unsafe_requests(source, edit):
    request = prepare(source, "request.json")
    edit(request)
    request["request_id"] = digest(request["request"])
    with pytest.raises(ValueError):
        validate_request(request)


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_json_is_strict(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_complete_bundle_and_failed_observations(source, tmp_path):
    request = prepared(source, repetitions=2)
    path, results = make_bundle(tmp_path / "good", request)
    decoded, entries = validate_bundle(path, results)
    assert decoded == request and len(entries) == 2
    path, results = make_bundle(tmp_path / "bad", request, failure=True)
    assert len(validate_bundle(path, results)[1]) == 2


@pytest.mark.parametrize(
    "edit",
    [
        lambda v: v["run"]["case"].update(rows=64),
        lambda v: v["records"][0]["metadata"].update(source_commit="f" * 40),
        lambda v: v["records"][0]["metadata"]["source_hashes"].update(
            {"torchcst/__init__.py": "f" * 64}
        ),
        lambda v: v["records"][0]["metadata"].update(gpu="NVIDIA T4"),
        lambda v: v["records"][0]["metadata"].update(torch="2.12.0"),
        lambda v: v.pop("execution_id"),
        lambda v: v.update(records=v["records"][:2]),
    ],
)
def test_changed_measurement_is_rejected_even_with_updated_transport_hash(
    source, tmp_path, edit
):
    path, results = make_bundle(tmp_path / "bundle", prepare(source, "request.json"))
    rewrite_result(results, edit)
    with pytest.raises(ValueError):
        validate_bundle(path, results)


def test_transport_hash_and_repetition_identity(source, tmp_path):
    path, results = make_bundle(tmp_path / "hash", prepare(source, "request.json"))
    rewrite_result(results, lambda v: v.update(extra="tampered"), refresh_hash=False)
    with pytest.raises(ValueError, match="artifact hash"):
        validate_bundle(path, results)
    path, results = make_bundle(tmp_path / "repeat", prepared(source, repetitions=2))
    bundle = decode((results / "submission.json").read_bytes())
    first = decode((results / bundle["jobs"][0]["artifact"]).read_bytes())[
        "execution_id"
    ]
    rewrite_result(results, lambda v: v.update(execution_id=first), 1)
    with pytest.raises(ValueError, match="independent execution"):
        validate_bundle(path, results)


def test_coordinator_consumes_frozen_snapshot_and_requires_hash(source, tmp_path):
    snapshot = tmp_path / "frozen.json"
    snapshot.write_bytes(
        encode(prepare(source, "request.json")["request"]["jobs"][0]["run"])
    )
    args = [
        sys.executable,
        "-m",
        "benchmarks.cuda.linear.run",
        "--snapshot",
        str(snapshot),
        "--validate-only",
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    result = subprocess.run(
        args + ["--snapshot-sha256", file_hash(snapshot)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert decode(result.stdout) == decode(snapshot.read_bytes())
    result = subprocess.run(
        args + ["--snapshot-sha256", "a" * 64], capture_output=True, text=True
    )
    assert result.returncode != 0


def test_pool_adapter_only_submits_to_default_queue_and_preserves_pending(
    source, tmp_path, monkeypatch
):
    request_file = tmp_path / "request.json"
    request_file.write_bytes(encode(prepare(source, "request.json")))
    calls = []

    def fake(pool, *args, **kwargs):
        calls.append(args)
        driver = Path(args[args.index("--script") + 1])
        assert "benchmarks.automation.driver import run" in driver.read_text()
        return SimpleNamespace(
            stdout=encode({"id": "l4job-fixture", "directory": "/fixture"}),
            returncode=0,
        )

    monkeypatch.setattr(colab, "pool_call", fake)
    pending = colab.submit(request_file, source, tmp_path / "pending", "/pool.py")
    assert len(pending["jobs"]) == 1
    assert "--state-root" not in calls[0] and calls[0][0] == "submit"
    assert decode((tmp_path / "pending/pending.json").read_bytes()) == pending


@pytest.mark.parametrize(
    "accelerator,name,capability",
    [
        ("T4", "Tesla T4", [7, 5]),
        ("L4", "NVIDIA L4", [8, 9]),
        ("A100", "NVIDIA A100-SXM4-40GB", [8, 0]),
        ("H100", "NVIDIA H100 80GB HBM3", [9, 0]),
        ("G4", "NVIDIA RTX PRO 6000 Blackwell Server Edition", [12, 0]),
    ],
)
def test_all_colab_gpu_types_bind_to_actual_hardware(
    source, tmp_path, accelerator, name, capability
):
    validate_hardware(accelerator, name, capability)
    request = prepared(source, targets=[accelerator])
    path, results = make_bundle(tmp_path / "bundle", request)
    assert len(validate_bundle(path, results)[1]) == 1
    rewrite_result(
        results,
        lambda v: v["records"][0]["metadata"].update(
            gpu="NVIDIA L4" if accelerator != "L4" else "Tesla T4"
        ),
    )
    with pytest.raises(ValueError, match="actual GPU"):
        validate_bundle(path, results)


def test_json_controls_target_matrix_without_cli_overrides(source):
    request = prepared(source, targets=["L4", "A100", "G4"], repetitions=2)
    assert len(request["request"]["jobs"]) == 6
    assert {j["accelerator"] for j in request["request"]["jobs"]} == {
        "L4",
        "A100",
        "G4",
    }
    with pytest.raises(ValueError, match="at most 6"):
        prepared(source, targets=["L4", "A100", "G4"], repetitions=3)
