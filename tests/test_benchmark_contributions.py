"""Admission boundaries between fork artifacts, trusted source and central DB."""

import copy
import io
import json
import stat
import subprocess
import zipfile

import pytest

from benchmarks.automation.contract import digest, prepare
from benchmarks.automation.contributions import (
    admit_run,
    extract_data,
    fetch,
    trusted_request,
    workflow_origin,
)
from benchmarks.automation.hosted import install_credentials, preserve_receipts
from tests.test_benchmark_automation import make_bundle, make_source


def metadata(commit):
    repository = {
        "id": 2,
        "full_name": "participant/torchcst",
        "fork": True,
        "private": False,
        "source": {"full_name": "takumiecd/torchcst"},
        "default_branch": "main",
    }
    run = {
        "id": 123,
        "repository": {"id": 2},
        "head_repository": {"id": 2},
        "event": "workflow_dispatch",
        "path": ".github/workflows/benchmark.yml",
        "status": "completed",
        "conclusion": "success",
        "head_branch": "main",
        "head_sha": commit,
        "actor": {"login": "participant"},
        "run_attempt": 1,
    }
    return repository, run


def zip_data(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


def accept_main(source):
    subprocess.run(
        ["git", "-C", str(source), "update-ref", "refs/remotes/origin/main", "HEAD"],
        check=True,
    )


@pytest.mark.parametrize(
    "edit",
    [
        lambda repo, run: repo.update(source={"full_name": "another/project"}),
        lambda repo, run: repo.update(private=True),
        lambda repo, run: run.update(event="pull_request"),
        lambda repo, run: run.update(path=".github/workflows/fake.yml"),
        lambda repo, run: run.update(head_branch="unreviewed"),
        lambda repo, run: run.update(head_repository={"id": 3}),
        lambda repo, run: run.update(status="in_progress"),
        lambda repo, run: run.update(conclusion="cancelled"),
    ],
)
def test_only_completed_public_measurement_runs_are_admitted(edit):
    repo, run = metadata("a" * 40)
    edit(repo, run)
    with pytest.raises(ValueError):
        admit_run(repo, run, "takumiecd/torchcst")


def test_provenance_is_stable_across_ingestion_retries():
    repo, run = metadata("a" * 40)
    origin = admit_run(repo, run, "takumiecd/torchcst")
    run["run_attempt"] = 2
    assert origin == workflow_origin(repo["full_name"], run)
    assert origin["workflow"]["GITHUB_ACTOR"] == "participant"


@pytest.mark.parametrize(
    "name",
    ["../outside.json", "/absolute.json", "results/../outside", "results\\evil.json"],
)
def test_zip_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        extract_data(zip_data({name: b"{}"}), tmp_path / "out")


def test_zip_links_are_rejected_and_scripts_never_extracted(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo("results/records/link.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../../secret")
    with pytest.raises(ValueError, match="unsafe"):
        extract_data(buffer.getvalue(), tmp_path / "link")
    extract_data(
        zip_data(
            {"pending/evil.py": b"raise Exception()", "results/submission.json": b"{}"}
        ),
        tmp_path / "safe",
    )
    assert list((tmp_path / "safe").rglob("*.py")) == []
    assert (tmp_path / "safe/results/submission.json").read_bytes() == b"{}"


def test_request_must_match_reviewed_upstream_source(tmp_path):
    source = make_source(tmp_path)
    accept_main(source)
    request = prepare(source, "request.json")
    trusted_request(source, request)
    forged = copy.deepcopy(request)
    forged["request"]["jobs"][0]["run"]["case"]["rows"] = 7
    forged["request_id"] = digest(forged["request"])
    with pytest.raises(ValueError, match="differs"):
        trusted_request(source, forged)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "unaccepted",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="not accepted"):
        trusted_request(source, prepare(source, "request.json"))


def test_fetch_binds_github_metadata_request_and_results(tmp_path):
    source = make_source(tmp_path)
    accept_main(source)
    request = prepare(source, "request.json")
    request_file, results = make_bundle(tmp_path / "bundle", request)
    repo, run = metadata(request["request"]["source_commit"])
    artifacts = [
        {
            "id": 11,
            "name": "benchmark-request",
            "workflow_run": {"head_sha": run["head_sha"]},
        },
        {
            "id": 12,
            "name": "benchmark-results",
            "workflow_run": {"head_sha": run["head_sha"]},
        },
    ]

    class FakeGitHub:
        def json(self, path):
            if path == "repos/participant/torchcst":
                return repo
            if path.endswith("/123"):
                return run
            return {"total_count": 2, "artifacts": artifacts}

        def archive(self, repository, artifact):
            if artifact["id"] == 11:
                return zip_data({"request.json": request_file.read_bytes()})
            return zip_data(
                {
                    "results/" + str(p.relative_to(results)): p.read_bytes()
                    for p in results.rglob("*.json")
                }
            )

    output = tmp_path / "fetched"
    result = fetch(
        repo["full_name"], 123, source, output, "takumiecd/torchcst", FakeGitHub()
    )
    assert result["request_id"] == request["request_id"]
    assert json.loads((output / "provenance.json").read_text()) == workflow_origin(
        repo["full_name"], run
    )
    assert (output / "measurement/results/submission.json").read_bytes() == (
        results / "submission.json"
    ).read_bytes()


def test_oauth_secret_stays_private_outside_checkout(tmp_path):
    token = {
        "refresh_token": "private",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client",
        "client_secret": "secret",
    }
    path = install_credentials(json.dumps(token), tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == token
    token["token_uri"] = "https://untrusted.invalid/token"
    with pytest.raises(ValueError, match="endpoint"):
        install_credentials(json.dumps(token), tmp_path / "other")


def test_receipts_never_export_live_session_credentials(tmp_path):
    root = tmp_path / ".local/state/colab-l4-pool"
    slot = root / "slots/1"
    slot.mkdir(parents=True)
    (slot / "state.json").write_text(
        json.dumps(
            {"status": "stopped", "session": "cst-pool-owned", "endpoint": "owned-id"}
        )
    )
    (slot / "sessions.json").write_text('{"token":"runtime-secret"}')
    job = root / "jobs/colabjob-fixture"
    job.mkdir(parents=True)
    (job / "spec.json").write_text('{"id":"job"}')
    (job / "transport.log").write_text("runtime-secret")
    output = tmp_path / "receipts"
    preserve_receipts(tmp_path, output)
    assert (output / "colabjob-fixture/spec.json").exists()
    assert not list(output.rglob("sessions.json"))
    assert not list(output.rglob("*.log"))
    assert all(
        b"runtime-secret" not in path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    )
