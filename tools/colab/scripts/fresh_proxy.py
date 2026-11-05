"""Bounded host-only retrieval using ephemeral credentials for an owned VM.

Run with the installed Colab CLI's Python. Never import common.state or open a
StateStore: even its read helpers can create lock files or prune sessions.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit


def require(condition):
    if not condition:
        raise RuntimeError("Fresh-proxy ownership or response validation failed")


def cleanup_code(spec):
    job = spec["id"]
    require(re.fullmatch(r"(?:l4job|colabjob)-[0-9a-f]{32}", job))
    require(spec["remote_archive"] == f"/content/{job}.tar.gz")
    require(spec["remote_result"] == f"/content/{job}-results.tar.gz")
    return (
        "import shutil\nfrom pathlib import Path\n"
        + f"shutil.rmtree({str('/content/' + job)!r})\n"
        + f"Path({spec['remote_archive']!r}).unlink()\n"
        + f"Path({spec['remote_result']!r}).unlink()\n"
        + f'print("POOL_CLEANED {job}")\n'
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cleanup_receipt(request, slot, spec):
    """Independently require the normal verified-result boundary before networking."""
    directory = Path(request["spec_file"]).parent
    archive = directory / "results.tar.gz"
    receipt_file = directory / "receipt.json"
    require(archive.is_file() and not archive.is_symlink())
    require(receipt_file.is_file() and not receipt_file.is_symlink())
    receipt = json.loads(receipt_file.read_text())
    require(
        type(receipt.get("slot")) is int
        and receipt["slot"] == request["slot"]
        and receipt.get("session") == slot["session"] == request["session"]
        and receipt.get("archive_sha256") == file_hash(archive)
    )
    results = directory / "results"
    require(results.is_dir() and not results.is_symlink())
    paths = list(results.rglob("*"))
    require(not any(path.is_symlink() for path in paths))
    inventory = {
        path.relative_to(results).as_posix(): path for path in paths if path.is_file()
    }
    manifest_file = inventory.pop("manifest.json", None)
    require(manifest_file is not None)
    manifest = json.loads(manifest_file.read_text())
    require(isinstance(manifest, dict) and set(manifest) == set(inventory))
    require(all(file_hash(path) == manifest[name] for name, path in inventory.items()))
    result = json.loads(inventory["result.json"].read_text())
    require(
        result["id"] == spec["id"] and result["source_sha256"] == spec["source_sha256"]
    )


def proxy_location(url):
    parsed = urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.port in (None, 443)
    )
    # Token renewal must not redirect retrieval to a different VM or path.
    return parsed.hostname, parsed.path.rstrip("/")


def fresh_session(cached, assignments, endpoint, accelerator):
    require(
        cached.endpoint == endpoint
        and cached.accelerator == accelerator
        and cached.variant == "GPU"
    )
    matches = [a for a in assignments if a.endpoint == endpoint]
    require(len(matches) == 1)
    assignment = matches[0]
    require(
        assignment.accelerator.value == accelerator and assignment.variant.value == 1
    )
    info = assignment.runtime_proxy_info
    require(
        isinstance(info.token, str)
        and bool(info.token)
        and info.token_expires_in_seconds > 60
        and proxy_location(info.url) == proxy_location(cached.url)
    )
    return cached.model_copy(update={"token": info.token, "url": info.url})


@contextlib.contextmanager
def bounded_contents_requests(requests, seconds):
    original = requests.request

    def request(method, url, **kwargs):
        require(method == "GET")
        kwargs["timeout"] = seconds
        kwargs["allow_redirects"] = False
        response = original(method, url, **kwargs)
        require(response.status_code == 200)
        return response

    requests.request = request
    try:
        yield
    finally:
        requests.request = original


def download(contents, remote, destination):
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink())
    temporary = destination.with_name(
        destination.name + "." + uuid.uuid4().hex + ".part"
    )
    try:
        contents.download(remote, str(temporary))
        require(temporary.is_file() and not temporary.is_symlink())
        # Download is complete; pool manifest/hash verification follows before
        # any receipt, job success or remote cleanup.
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def cleanup(runtime_factory, session, spec):
    require(session.kernel_id and session.session_id)
    runtime = runtime_factory(
        session.url,
        session.token,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
    )
    try:
        outputs = runtime.execute_code(cleanup_code(spec), timeout=30)
        require(not any(o.get("output_type") == "error" for o in outputs))
        text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        require("POOL_CLEANED " + spec["id"] in text.splitlines())
    finally:
        # Close channels only. Never shut down, restart or allocate a kernel.
        runtime.stop(shutdown_kernel=False)


def sdk_operation(request):
    import requests
    from colab_cli.client import Client, Prod
    from colab_cli.contents import ContentsClient
    from colab_cli.runtime import ColabRuntime
    from colab_cli.state import SessionState
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    require(request["operation"] in ("download", "cleanup"))
    require(type(request["slot"]) is int and request["slot"] > 0)
    require(Path(request["slot_file"]).parent.name == str(request["slot"]))
    require(
        Path(request["destination"])
        == Path(request["spec_file"]).parent / "results.tar.gz"
    )
    slot = json.loads(Path(request["slot_file"]).read_text())
    spec = json.loads(Path(request["spec_file"]).read_text())
    require(
        slot["session"] == request["session"]
        and slot["endpoint"] == request["endpoint"]
        and slot["job"] == spec["id"]
        and slot["status"] == "ready"
        and spec["accelerator"] == request["accelerator"]
    )
    cleanup_code(spec)  # Validate all remote paths before any network access.
    if request["operation"] == "cleanup":
        verify_cleanup_receipt(request, slot, spec)
    records = json.loads(Path(request["session_file"]).read_text())
    cached = SessionState(**records[request["session"]])
    require(
        cached.name == request["session"] and cached.endpoint == request["endpoint"]
    )
    token_file = Path.home() / ".config/colab-cli/token.json"
    # AuthorizedSession refreshes these credentials only in memory. Calling
    # colab_cli.auth.get_credentials would write refreshed OAuth tokens to disk.
    credentials = Credentials.from_authorized_user_file(str(token_file))
    with AuthorizedSession(credentials) as authorized:
        original_request = authorized.request

        def bounded_request(method, url, **kwargs):
            kwargs["timeout"] = 30
            kwargs["allow_redirects"] = False
            return original_request(method, url, **kwargs)

        authorized.request = bounded_request
        response = authorized.get("https://www.googleapis.com/oauth2/v2/userinfo")
        require(response.status_code == 200)
        identity = response.json()
        require(
            identity.get("email") == request["account"]
            and identity.get("verified_email") is True
        )
        assignments = Client(Prod(), authorized).list_assignments()
        session = fresh_session(
            cached, assignments, request["endpoint"], request["accelerator"]
        )
        # Detect ownership/state changes during identity and assignment reads.
        require(json.loads(Path(request["slot_file"]).read_text()) == slot)
        if request["operation"] == "download":
            with bounded_contents_requests(requests, 840):
                download(
                    ContentsClient(session),
                    spec["remote_result"],
                    request["destination"],
                )
        else:
            cleanup(ColabRuntime, session, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-stdin", action="store_true", required=True)
    parser.parse_args()
    # SDK errors may include token-bearing URLs, request headers or response
    # bodies. Never print them, even on failure or timeout.
    logging.disable(logging.CRITICAL)
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            sdk_operation(json.load(sys.stdin))
    except BaseException:  # noqa: BLE001 - SDK messages can contain credentials.
        print(
            "Fresh-proxy transport failed; remote state is uncertain", file=sys.stderr
        )
        return 1
    print("FRESH_PROXY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
