"""Import data from an Actions run, without executing contributor source."""

import argparse
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from benchmarks.automation.contract import (
    decode,
    encode,
    git,
    load_request,
    prepare,
    relative_path,
    sha,
)
from benchmarks.automation.ingest import validate_bundle

MAX_ARCHIVE = 64 * 1024**2
MAX_EXPANDED = 128 * 1024**2
WORKFLOW = ".github/workflows/benchmark.yml"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, token):
        self.token = token

    def request(self, path):
        return urllib.request.Request(
            "https://api.github.com/" + path,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def json(self, path):
        with urllib.request.urlopen(self.request(path), timeout=60) as response:
            raw = response.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise ValueError("oversized GitHub metadata")
        return decode(raw)

    def archive(self, repository, artifact):
        if artifact["expired"] or artifact["size_in_bytes"] > MAX_ARCHIVE:
            raise ValueError("expired or oversized Actions artifact")
        artifact_id = artifact["id"]
        if type(artifact_id) is not int or artifact_id <= 0:
            raise ValueError("invalid Actions artifact ID")
        request = self.request(
            f"repos/{repository}/actions/artifacts/{artifact_id}/zip"
        )
        try:
            urllib.request.build_opener(NoRedirect()).open(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise RuntimeError(
                    "cannot download Actions artifact; configure BENCHMARK_ARTIFACT_READ_TOKEN with read access to the submitting repository"
                ) from None
            location = error.headers["Location"]
        else:
            raise ValueError("GitHub did not return an artifact download redirect")
        if not location.startswith("https://"):
            raise ValueError("unsafe artifact redirect")
        # The signed storage URL receives no GitHub credential.
        with urllib.request.urlopen(location, timeout=60) as response:
            raw = response.read(MAX_ARCHIVE + 1)
        if len(raw) > MAX_ARCHIVE:
            raise ValueError("oversized Actions artifact")
        expected = artifact.get("digest", "")
        if expected != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ValueError("Actions artifact digest mismatch")
        return raw


def repository_name(value):
    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value
    ):
        raise ValueError("expected owner/repository")
    return value


def admit_run(repository, run, upstream):
    """GitHub metadata, rather than an uploaded provenance.json, is authoritative."""
    repository_name(upstream)
    name = repository_name(repository["full_name"])
    if name.lower() != upstream.lower() and (
        not repository.get("fork")
        or repository.get("source", {}).get("full_name", "").lower() != upstream.lower()
    ):
        raise ValueError(
            "submission must be from the upstream repository or a public fork"
        )
    if repository.get("private"):
        raise ValueError("only public submissions are supported")
    if (
        run["repository"]["id"] != repository["id"]
        or run["head_repository"]["id"] != repository["id"]
        or run["event"] != "workflow_dispatch"
        or run["path"] != WORKFLOW
        or run["status"] != "completed"
        or run["conclusion"] not in ("success", "failure")
        or run["head_branch"] != repository["default_branch"]
    ):
        raise ValueError(
            "submission is not a completed measurement workflow on its default branch"
        )
    sha(run["head_sha"], 40)
    return workflow_origin(name, run)


def workflow_origin(repository, run, *, hosted=True):
    return {
        "origin": "github-actions-hosted-colab"
        if hosted
        else "github-actions-colab-bridge",
        "workflow": {
            "GITHUB_REPOSITORY": repository,
            "GITHUB_RUN_ID": str(run["id"]),
            "GITHUB_SHA": run["head_sha"],
            "GITHUB_ACTOR": run["actor"]["login"],
            "GITHUB_WORKFLOW_REF": f"{repository}/{WORKFLOW}@refs/heads/{run['head_branch']}",
        },
        "certification": "not assessed",
    }


def extract_data(raw, directory, *, request=False):
    """Bound zip expansion; extract JSON data only, never scripts or links."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist()) > 200:
            raise ValueError("too many artifact entries")
        for entry in archive.infolist():
            name = relative_path(
                entry.filename.rstrip("/") if entry.is_dir() else entry.filename
            )
            mode = entry.external_attr >> 16
            total += entry.file_size
            if (
                name in seen
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
                or entry.file_size > 32 * 1024**2
                or total > MAX_EXPANDED
            ):
                raise ValueError("unsafe or oversized artifact entry")
            seen.add(name)
            if entry.is_dir():
                continue
            allowed = (
                name == "request.json"
                if request
                else (
                    name == "results/submission.json"
                    or re.fullmatch(r"results/records/[a-z][a-z0-9_.-]*\.json", name)
                )
            )
            if allowed:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)


def trusted_request(root, request):
    """Recreate the request using reviewed upstream data and central Python code."""
    commit = request["request"]["source_commit"]
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "origin/main"],
        capture_output=True,
        check=False,
    ).returncode:
        raise ValueError("measured commit is not accepted upstream main source")
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(root),
                directory,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", directory, "checkout", "--quiet", "--detach", commit],
            check=True,
            capture_output=True,
        )
        expected = prepare(directory, request["request"]["configuration"]["path"])
    if expected != request:
        raise ValueError(
            "submitted request differs from the accepted source/measurement JSON"
        )


def fetch(repository, run_id, root, output, upstream, github):
    repository_name(repository)
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("invalid workflow run ID")
    metadata = github.json(f"repos/{repository}")
    run = github.json(f"repos/{repository}/actions/runs/{run_id}")
    provenance = admit_run(metadata, run, upstream)
    listing = github.json(
        f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100"
    )
    if listing["total_count"] > 100:
        raise ValueError("too many workflow artifacts")
    artifacts = {}
    for entry in listing["artifacts"]:
        if entry["name"] in ("benchmark-request", "benchmark-results"):
            if entry["name"] in artifacts:
                raise ValueError(
                    "ambiguous measurement artifact; submit an unrepeated workflow run"
                )
            if entry["workflow_run"]["head_sha"] != run["head_sha"]:
                raise ValueError("artifact source commit mismatch")
            artifacts[entry["name"]] = entry
    if set(artifacts) != {"benchmark-request", "benchmark-results"}:
        raise ValueError("measurement artifacts missing")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    extract_data(
        github.archive(repository, artifacts["benchmark-request"]),
        output / "request",
        request=True,
    )
    request_file = output / "request/request.json"
    request = load_request(request_file)
    if request["request"]["source_commit"] != run["head_sha"]:
        raise ValueError("request does not belong to workflow source commit")
    trusted_request(root, request)
    # Preserve the original origin of historical Mac-bridge observations.
    # Their immutable DB provenance must survive import/retry unchanged.
    provenance = workflow_origin(
        metadata["full_name"],
        run,
        hosted="benchmarks/automation/hosted.py" in request["request"]["source_files"],
    )
    extract_data(
        github.archive(repository, artifacts["benchmark-results"]),
        output / "measurement",
    )
    validate_bundle(request_file, output / "measurement/results")
    (output / "provenance.json").write_bytes(encode(provenance))
    return {
        "repository": repository,
        "run_id": run_id,
        "request_id": request["request_id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "provenance"])
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        github = GitHub(os.environ["GH_TOKEN"])
        if args.command == "fetch":
            result = fetch(
                args.repository,
                args.run_id,
                args.source,
                args.output,
                os.environ["GITHUB_REPOSITORY"],
                github,
            )
        else:
            repository_name(args.repository)
            run = github.json(f"repos/{args.repository}/actions/runs/{args.run_id}")
            if run["head_sha"] != git(args.source, "rev-parse", "HEAD"):
                raise ValueError("workflow and checkout commit differ")
            args.output.write_bytes(encode(workflow_origin(args.repository, run)))
            result = {"provenance": "saved"}
        print(encode(result).decode(), end="")
        return 0
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - do not expose signed URLs or credentials
        print(
            "contribution retrieval failed; inspect metadata and read-token access privately",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
