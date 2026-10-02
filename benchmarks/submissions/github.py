"""Bounded GitHub metadata/attachment transport; never execute submitted files."""

import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

from benchmarks.submissions.policy import account_login, positive_id
from torchcst._backends.cuda.serialization import decode_json

TITLE_PREFIX = "[Benchmark result] "
API_LIMIT = 1024**2


def attachment_url(value):
    if type(value) is not str or len(value) > 2048:
        raise ValueError("invalid attachment URL")
    url = urllib.parse.urlsplit(value)
    path = urllib.parse.unquote(url.path)
    if (
        url.scheme != "https"
        or url.netloc != "github.com"
        or url.query
        or url.fragment
        or not re.fullmatch(
            r"/user-attachments/files/[1-9][0-9]*/[^/\\\x00-\x20]+\.json", path
        )
        or "/../" in path
        or "/./" in path
    ):
        raise ValueError("attach one JSON file hosted by GitHub user-attachments")
    return value


def redirect_url(value):
    url = urllib.parse.urlsplit(value)
    if (
        url.scheme != "https"
        or url.username
        or url.password
        or url.port not in (None, 443)
    ):
        raise ValueError("unsafe attachment redirect")
    if url.hostname == "github.com":
        if not url.path.startswith("/user-attachments/files/"):
            raise ValueError("unsafe GitHub attachment path")
    elif url.hostname != "objects.githubusercontent.com":
        raise ValueError("attachment redirected outside GitHub storage")
    return value


class AttachmentRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirect_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class Submission:
    repository: str
    issue_id: int
    issue_number: int
    submitter_id: int
    submitter_login: str
    issue_body_sha256: str
    artifact_url: str

    def validate(self, policy):
        if self.repository != policy.repository:
            raise ValueError("submission must belong to the policy repository")
        for value in (self.issue_id, self.issue_number, self.submitter_id):
            positive_id(value)
        account_login(self.submitter_login)
        if not re.fullmatch(r"[0-9a-f]{64}", self.issue_body_sha256):
            raise ValueError("invalid issue body hash")
        attachment_url(self.artifact_url)
        return self

    def declaration(self):
        return asdict(self)


def submission_from_issue(issue, policy):
    if "pull_request" in issue:
        raise ValueError("submit benchmark results as an issue, not a PR")
    if type(issue.get("title")) is not str or not issue["title"].startswith(
        TITLE_PREFIX
    ):
        raise ValueError("use the benchmark result issue form")
    user = issue["user"]
    if user.get("type") != "User":
        raise ValueError("submission needs a GitHub user account")
    body = issue.get("body")
    if type(body) is not str or len(body.encode()) > 65536:
        raise ValueError("missing or oversized issue body")
    # Only GitHub attachment links count; external download links are rejected.
    urls = re.findall(r"https://[^\s<>\"\)]+", body)
    if len(urls) != 1:
        raise ValueError("attach exactly one completed benchmark JSON per issue")
    url = attachment_url(urls[0])
    expected = f"https://github.com/{policy.repository}/issues/{issue['number']}"
    if issue.get("html_url") != expected:
        raise ValueError("unexpected issue origin")
    return Submission(
        policy.repository,
        issue["id"],
        issue["number"],
        user["id"],
        user["login"],
        hashlib.sha256(body.encode()).hexdigest(),
        url,
    ).validate(policy)


class GitHub:
    def __init__(self, token):
        if not token:
            raise ValueError("GitHub token is missing")
        self.token = token

    def api(self, path, *, method="GET", value=None):
        import json

        data = None if value is None else json.dumps(value).encode()
        request = urllib.request.Request(
            "https://api.github.com/" + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        # API calls must not redirect a credential to any other host.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                raise ValueError("unexpected GitHub API redirect")

        with urllib.request.build_opener(NoRedirect()).open(
            request, timeout=30
        ) as response:
            raw = response.read(API_LIMIT + 1)
        if len(raw) > API_LIMIT:
            raise ValueError("oversized GitHub API response")
        return decode_json(raw)

    def issue(self, repository, number):
        positive_id(number)
        return self.api(f"repos/{repository}/issues/{number}")

    def attachment(self, url, limit):
        attachment_url(url)
        # Public issue attachments require no token. Never forward GH credentials.
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.build_opener(AttachmentRedirect()).open(
            request, timeout=30
        ) as response:
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("compressed HTTP attachments are not supported")
            raw = response.read(limit + 1)
        if not raw or len(raw) > limit:
            raise ValueError("empty or oversized result JSON")
        return raw

    def comment(self, repository, number, body):
        return self.api(
            f"repos/{repository}/issues/{number}/comments",
            method="POST",
            value={"body": body},
        )

    def close(self, repository, number):
        return self.api(
            f"repos/{repository}/issues/{number}",
            method="PATCH",
            value={"state": "closed", "state_reason": "completed"},
        )
