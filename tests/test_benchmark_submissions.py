"""Two-tier identity, data-only transport and bounded issue intake."""

import copy
import json
import os
import subprocess
import sys
import uuid
from argparse import Namespace

import pytest

from benchmarks.submissions.__main__ import (
    fetch,
    issue_number,
    load_fetched,
    validate_result,
    write_feedback,
)
from benchmarks.submissions.github import (
    attachment_url,
    redirect_url,
    submission_from_issue,
)
from benchmarks.submissions.policy import Policy, load_policy
from tests.benchmark_database_fixtures import artifact, raw_artifact


@pytest.fixture
def policy():
    return load_policy()


def recognized(policy, user_id=123):
    value = copy.deepcopy(policy.declaration)
    value["recognized"] = [
        {
            "github_user_id": user_id,
            "login": "recognized-user",
            "reason": "Administrator-approved submissions",
        }
    ]
    return Policy(value)


def issue(policy, *, number=7, user_id=123):
    return {
        "id": 9000 + number,
        "number": number,
        "title": "[Benchmark result] local GPU",
        "html_url": f"https://github.com/{policy.repository}/issues/{number}",
        "body": "### Benchmark JSON\n\n[result.json](https://github.com/user-attachments/files/123/result.json)",
        "user": {"id": user_id, "login": "example-user", "type": "User"},
    }


def new_run():
    return raw_artifact(
        artifact()
        | {"execution_id": str(uuid.uuid4()), "started_at": "2026-10-02T00:00:00+00:00"}
    )


def test_general_needs_no_registration_and_points_start_at_one(policy):
    assert policy.tier(123) == "general"
    assert policy.trust_points(123) == 1
    assert policy.daily_limit(123) == 10
    trusted = recognized(policy)
    assert trusted.tier(123) == "recognized"
    assert trusted.daily_limit(123) is None
    assert trusted.trust_points(123) == 10
    assert trusted.tier(456) == "general"


def test_recognition_uses_numeric_id_and_points_are_independent(policy):
    value = copy.deepcopy(recognized(policy).declaration)
    value["recognized"][0]["trust_points"] = 1
    changed = Policy(value)
    assert changed.tier(123) == "recognized" and changed.daily_limit(123) is None
    assert changed.trust_points(123) == 1
    account = issue(changed)
    account["user"]["login"] = "renamed-user"
    submitted = submission_from_issue(account, changed)
    assert changed.tier(submitted.submitter_id) == "recognized"
    assert submitted.submitter_login == "renamed-user"
    assert changed.tier(456) == "general"  # A copied display name grants nothing.


@pytest.mark.parametrize(
    "field,value",
    [
        ("general_daily_runs", 0),
        ("general_daily_runs", True),
        ("general_trust_points", 0),
        ("recognized_trust_points", -1),
        ("max_file_bytes", 26 * 1024**2),
        ("recognized", {}),
        ("schema_version", True),
    ],
)
def test_invalid_policy_is_rejected(policy, field, value):
    declaration = copy.deepcopy(policy.declaration)
    declaration[field] = value
    with pytest.raises(ValueError):
        Policy(declaration)


def test_duplicate_recognized_ids_and_unknown_fields_rejected(policy):
    declaration = copy.deepcopy(recognized(policy).declaration)
    declaration["recognized"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        Policy(declaration)
    declaration = policy.declaration | {"trust_algorithm": "arbitrary"}
    with pytest.raises(ValueError):
        Policy(declaration)


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/user-attachments/files/123/result.json",
        "https://github.com.evil.test/user-attachments/files/123/result.json",
        "https://github.com@evil.test/user-attachments/files/123/result.json",
        "https://github.com/user-attachments/files/123/../result.json",
        "https://github.com/user-attachments/files/123/%2Fresult.json",
        "https://github.com/user-attachments/files/123/result.zip",
        "https://github.com/user-attachments/files/123/result.json?token=secret",
        "https://127.0.0.1/result.json",
    ],
)
def test_only_github_json_attachments_accepted(url):
    with pytest.raises(ValueError):
        attachment_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://objects.githubusercontent.com/file",
        "https://objects.githubusercontent.com.evil.test/file",
        "https://github.com/login",
        "https://127.0.0.1/file",
        "https://github.com:444/user-attachments/files/1/a",
    ],
)
def test_redirects_cannot_reach_other_hosts(url):
    with pytest.raises(ValueError):
        redirect_url(url)


def test_github_storage_redirect_can_keep_signed_query():
    assert redirect_url(
        "https://objects.githubusercontent.com/storage/result.json?signature=opaque"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"pull_request": {}},
        {"title": "Normal issue"},
        {"html_url": "https://github.com/other/project/issues/7"},
        {"body": "No attachment"},
        {"body": "https://other.test/result.json"},
        {
            "body": "https://github.com/user-attachments/files/1/a.json https://github.com/user-attachments/files/2/b.json"
        },
    ],
)
def test_non_submission_issues_rejected(policy, change):
    with pytest.raises(ValueError):
        submission_from_issue(issue(policy) | change, policy)


def test_event_uses_issue_author_not_a_maintainer_editor(policy, tmp_path, monkeypatch):
    event = {
        "action": "edited",
        "repository": {"full_name": policy.repository},
        "issue": issue(policy),
        "sender": {"id": 456},
    }
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    with pytest.raises(ValueError, match="their author"):
        issue_number(Namespace(issue_number="7"), policy)
    event["sender"]["id"] = 123
    path.write_text(json.dumps(event))
    assert issue_number(Namespace(issue_number="7"), policy) == 7
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    assert issue_number(Namespace(issue_number="7"), policy) == 7


def test_uploaded_json_cannot_assign_identity_or_points(policy):
    raw = raw_artifact(
        json.loads(new_run())
        | {"submitter_id": 999, "trust_points": 1000000, "tier": "recognized"}
    )
    validate_result(raw, policy)
    submitted = submission_from_issue(issue(policy), policy)
    assert (
        submitted.submitter_id == 123
        and policy.trust_points(submitted.submitter_id) == 1
    )


def test_fetch_revalidation_detects_modified_data(policy, tmp_path):
    class API:
        def issue(self, repository, number):
            return issue(policy, number=number)

        def attachment(self, url, limit):
            return new_run()

    directory = tmp_path / "fetched"
    fetch(API(), 7, policy, directory)
    raw, submitted = load_fetched(directory, policy)
    assert (
        submitted.submitter_login == "example-user" and json.loads(raw)["execution_id"]
    )
    (directory / "result.json").write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="changed"):
        load_fetched(directory, policy)


def test_new_submission_requires_uuid_and_consistent_benchmark(policy):
    with pytest.raises(ValueError, match="UUID"):
        validate_result(raw_artifact(), policy)
    value = json.loads(new_run())
    value["records"][2]["result"]["graph"]["median_ms"] = 100
    with pytest.raises(ValueError, match="median"):
        validate_result(raw_artifact(value), policy)
    with pytest.raises(ValueError):
        validate_result(b'{"status":"PASS","status":"FAIL"}', policy)


def test_local_preflight_needs_no_github_or_database_login(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(new_run())
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("GITHUB_REPOSITORY", "DATABASE_URL", "GH_TOKEN")
    }
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.submissions", "check", str(path)],
        env=env,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["validated"] is True


def test_attachment_never_receives_github_token_and_is_bounded(monkeypatch):
    from benchmarks.submissions.github import GitHub

    seen = []

    class Response:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b"x" * limit

    class Opener:
        def open(self, request, timeout):
            seen.append(dict(request.header_items()))
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(ValueError, match="oversized"):
        GitHub("private-github-token").attachment(
            "https://github.com/user-attachments/files/1/result.json", 10
        )
    assert not any(k.lower() == "authorization" for k in seen[0])
    assert "private-github-token" not in str(seen)


def test_feedback_does_not_execute_or_echo_uploaded_source(policy):
    calls = []

    class API:
        def issue(self, repository, number):
            return issue(policy, number=number)

        def comment(self, *args):
            calls.append(("comment", args))

        def close(self, *args):
            calls.append(("close", args))

    receipt = {
        "run_id": "a" * 64,
        "submission_inserted": True,
        "submitter_login": "example-user",
        "tier": "general",
        "trust_points": 1,
    }
    write_feedback(API(), 7, policy, receipt=receipt)
    assert [c[0] for c in calls] == ["comment", "close"]
    assert "general" in calls[0][1][2] and "@example-user" in calls[0][1][2]
