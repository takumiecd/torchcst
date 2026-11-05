"""Ephemeral retrieval guards; no Colab credentials, GPU or network required."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tools.colab.scripts import fresh_proxy as transport
from tools.colab.scripts.pool import Pool

JOB = "l4job-" + "a" * 32
SPEC = {
    "id": JOB,
    "accelerator": "L4",
    "remote_archive": f"/content/{JOB}.tar.gz",
    "remote_result": f"/content/{JOB}-results.tar.gz",
}


class Session(NS):
    def model_copy(self, *, update):
        values = dict(vars(self))
        values.update(update)
        return Session(**values)


def cached():
    return Session(
        name="cst-pool-owned-1",
        endpoint="owned",
        accelerator="L4",
        variant="GPU",
        token="OLD-SECRET",
        url="https://proxy.example/vm",
        kernel_id="kernel",
        session_id="session",
    )


def assignment(**changes):
    values = {
        "endpoint": "owned",
        "accelerator": NS(value="L4"),
        "variant": NS(value=1),
        "runtime_proxy_info": NS(
            token="NEW-SECRET",
            token_expires_in_seconds=3600,
            url="https://proxy.example/vm/",
        ),
    }
    values.update(changes)
    return NS(**values)


@pytest.mark.parametrize(
    "change",
    [
        "absent",
        "ambiguous",
        "endpoint",
        "gpu",
        "cached-variant",
        "listed-variant",
        "origin",
        "path",
        "expiry",
        "userinfo",
        "query",
    ],
)
def test_rejects_unowned_or_changed_assignment(change):
    current, assignments = cached(), [assignment()]
    if change == "absent":
        assignments = []
    elif change == "ambiguous":
        assignments *= 2
    elif change == "endpoint":
        current.endpoint = "other"
    elif change == "gpu":
        assignments[0].accelerator.value = "A100"
    elif change == "cached-variant":
        current.variant = "TPU"
    elif change == "listed-variant":
        assignments[0].variant.value = 2
    elif change == "origin":
        assignments[0].runtime_proxy_info.url = "https://other.example/vm"
    elif change == "path":
        assignments[0].runtime_proxy_info.url += "-other"
    elif change == "expiry":
        assignments[0].runtime_proxy_info.token_expires_in_seconds = 1
    elif change == "userinfo":
        assignments[0].runtime_proxy_info.url = "https://user:pass@proxy.example/vm"
    else:
        assignments[0].runtime_proxy_info.url += "?token=SECRET"
    with pytest.raises(RuntimeError):
        transport.fresh_session(current, assignments, "owned", "L4")


def test_refresh_changes_only_memory_copy():
    current = cached()
    before = copy.deepcopy(vars(current))
    fresh = transport.fresh_session(current, [assignment()], "owned", "L4")
    assert vars(current) == before
    assert fresh.token == "NEW-SECRET" and fresh.kernel_id == current.kernel_id


def test_download_atomic_and_failure_retains_no_partial(tmp_path):
    class Contents:
        def download(self, remote, local):
            assert remote == SPEC["remote_result"]
            Path(local).write_bytes(b"archive")

    target = tmp_path / "results.tar.gz"
    transport.download(Contents(), SPEC["remote_result"], target)
    assert target.read_bytes() == b"archive" and len(list(tmp_path.iterdir())) == 1
    with pytest.raises(RuntimeError):
        transport.download(Contents(), SPEC["remote_result"], target)
    target.unlink()

    class Failed(Contents):
        def download(self, remote, local):
            Path(local).write_bytes(b"partial")
            raise TimeoutError("SECRET-URL")

    with pytest.raises(TimeoutError):
        transport.download(Failed(), SPEC["remote_result"], target)
    assert not list(tmp_path.iterdir())


def test_contents_timeout_redirect_and_method_boundaries():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return NS(status_code=200)

    requests = NS(request=request)
    with transport.bounded_contents_requests(requests, 840):
        requests.request("GET", "https://proxy.example/vm")
        with pytest.raises(RuntimeError):
            requests.request("DELETE", "https://proxy.example/vm")
    assert requests.request is request
    assert calls[0][2] == {"timeout": 840, "allow_redirects": False}


@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "missing-ack"])
def test_cleanup_exact_code_no_callbacks_no_kernel_shutdown(outcome):
    events = []

    class Runtime:
        def __init__(self, url, token, **kwargs):
            assert kwargs == {"kernel_id": "kernel", "session_id": "session"}

        def execute_code(self, code, **kwargs):
            assert code == transport.cleanup_code(SPEC) and kwargs == {"timeout": 30}
            events.append("execute")
            if outcome == "timeout":
                raise TimeoutError("SECRET")
            if outcome == "error":
                return [{"output_type": "error", "evalue": "SECRET"}]
            return [
                {
                    "output_type": "stream",
                    "text": "POOL_CLEANED " + JOB + "\n"
                    if outcome == "success"
                    else "other\n",
                }
            ]

        def stop(self, **kwargs):
            events.append(kwargs)

    if outcome == "success":
        transport.cleanup(Runtime, cached(), SPEC)
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            transport.cleanup(Runtime, cached(), SPEC)
    assert events == ["execute", {"shutdown_kernel": False}]


def install_fake_sdk(
    monkeypatch, tmp_path, *, email="owner@example.com", altered_slot=False
):
    """Exercise real helper orchestration, with imports exposing no write APIs."""
    session = cached()
    slot = {"status": "ready", "session": session.name, "endpoint": "owned", "job": JOB}
    paths = {}
    for name, data in [
        ("slot", slot),
        ("sessions", {session.name: vars(session)}),
        ("spec", SPEC),
        ("oauth", {"secret": "NOT-USED"}),
    ]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data))
        paths[name] = path
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    events = []

    class Authorized:
        def __init__(self, credentials):
            events.append("authorize-in-memory")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method, url, **kwargs):
            assert kwargs["timeout"] == 30 and kwargs["allow_redirects"] is False
            return NS(
                status_code=200, json=lambda: {"email": email, "verified_email": True}
            )

        def get(self, url):
            return self.request("GET", url)

    class Client:
        def __init__(self, env, authorized):
            pass

        def list_assignments(self):
            events.append("list")
            if altered_slot:
                paths["slot"].write_text("{}")
            return [assignment()]

    class Contents:
        def __init__(self, s):
            assert s.token == "NEW-SECRET"

        def download(self, remote, local):
            events.append("download")
            Path(local).write_bytes(b"archive")

    modules = {
        "requests": {"request": lambda *a, **k: None},
        "colab_cli.client": {"Client": Client, "Prod": lambda: object()},
        "colab_cli.contents": {"ContentsClient": Contents},
        "colab_cli.runtime": {"ColabRuntime": object},
        "colab_cli.state": {"SessionState": Session},
        "google.auth.transport.requests": {"AuthorizedSession": Authorized},
        "google.oauth2.credentials": {
            "Credentials": NS(from_authorized_user_file=lambda p: object())
        },
    }
    for name, values in modules.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    request = {
        "operation": "download",
        "slot_file": str(paths["slot"]),
        "session_file": str(paths["sessions"]),
        "spec_file": str(paths["spec"]),
        "session": session.name,
        "endpoint": "owned",
        "accelerator": "L4",
        "account": "owner@example.com",
        "destination": str(tmp_path / "results.tar.gz"),
    }
    return request, before, paths, events


@pytest.mark.parametrize(
    "email,changed",
    [
        ("owner@example.com", False),
        ("other@example.com", False),
        ("owner@example.com", True),
    ],
)
def test_sdk_orchestration_identity_ownership_and_no_cache_writes(
    tmp_path, monkeypatch, email, changed
):
    request, before, paths, events = install_fake_sdk(
        monkeypatch, tmp_path, email=email, altered_slot=changed
    )
    if email != "owner@example.com" or changed:
        with pytest.raises(RuntimeError):
            transport.sdk_operation(request)
        assert "download" not in events
    else:
        transport.sdk_operation(request)
        assert events == ["authorize-in-memory", "list", "download"]
        assert {p.name: p.read_bytes() for p in paths.values()} == before
        assert {p.name for p in tmp_path.iterdir()} == set(before) | {"results.tar.gz"}


def test_child_error_never_prints_sdk_secrets(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fresh_proxy", "--request-stdin"])
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("{}"))

    def fail(request):
        print("SECRET-STDOUT")
        print("SECRET-STDERR", file=sys.stderr)
        raise RuntimeError("token=NEW-SECRET body=PRIVATE")

    monkeypatch.setattr(transport, "sdk_operation", fail)
    assert transport.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Fresh-proxy transport failed; remote state is uncertain\n"


@pytest.mark.parametrize("outcome", ["success", "timeout", "secret-error"])
def test_parent_bounds_helper_uses_cli_python_and_hides_output(
    tmp_path, monkeypatch, outcome
):
    cli = tmp_path / "colab"
    cli.write_text(f"#!{sys.executable}\nfrom colab_cli.cli import main\n")
    pool = Pool(tmp_path / "state", cli=cli)
    pool.expected_account = "owner@example.com"
    pool.save_slot(1, status="ready", session="owned", endpoint="endpoint", job=JOB)
    pool.jobdir(JOB).mkdir()

    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert (
            kwargs["timeout"] == 900 and kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        )
        assert json.loads(kwargs["input"])["endpoint"] == "endpoint"
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("SECRET", 900, output="NEW-SECRET")
        return NS(
            returncode=0 if outcome == "success" else 1,
            stdout="FRESH_PROXY_OK\n" if outcome == "success" else "NEW-SECRET",
            stderr="PRIVATE",
        )

    monkeypatch.setattr(subprocess, "run", run)
    before = (pool.slotdir(1) / "state.json").read_bytes()
    if outcome == "success":
        pool.fresh_transport(1, JOB, "owned", "download")
    else:
        with pytest.raises(RuntimeError, match="quarantined") as error:
            pool.fresh_transport(1, JOB, "owned", "download")
        assert "SECRET" not in str(error.value) and error.value.__cause__ is None
    assert (pool.slotdir(1) / "state.json").read_bytes() == before


@pytest.mark.parametrize(
    "state_change", ["session", "endpoint", "job", "status", "account"]
)
def test_parent_ownership_guard_does_not_launch_or_mutate(
    tmp_path, monkeypatch, state_change
):
    pool = Pool(tmp_path / "state")
    pool.expected_account = "owner@example.com"
    state = {"status": "ready", "session": "owned", "endpoint": "endpoint", "job": JOB}
    if state_change == "account":
        pool.expected_account = None
    else:
        state[state_change] = "other" if state_change != "endpoint" else None
    pool.save_slot(1, **state)
    before = (pool.slotdir(1) / "state.json").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="ownership"):
        pool.fresh_transport(1, JOB, "owned", "download")
    assert (pool.slotdir(1) / "state.json").read_bytes() == before


def test_missing_sdk_interpreter_is_fail_closed(tmp_path):
    cli = tmp_path / "colab"
    cli.write_text("#!/not-installed/python\nfrom colab_cli.cli import main\n")
    pool = Pool(tmp_path / "state", cli=cli)
    pool.expected_account = "owner@example.com"
    pool.save_slot(1, status="ready", session="owned", endpoint="endpoint", job=JOB)
    with pytest.raises(RuntimeError, match="interpreter"):
        pool.fresh_transport(1, JOB, "owned", "download")


def test_retrieval_cannot_overwrite_session_cache(tmp_path, monkeypatch):
    request, before, paths, events = install_fake_sdk(monkeypatch, tmp_path)
    request["destination"] = str(paths["sessions"])
    with pytest.raises(RuntimeError):
        transport.sdk_operation(request)
    assert events == []
    assert {p.name: p.read_bytes() for p in paths.values()} == before
