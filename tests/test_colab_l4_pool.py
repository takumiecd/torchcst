"""Scheduler invariants; these tests never allocate GPUs or access Colab."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "tools/colab-l4-pool/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pool_module = load("pool")
remote = load("remote_runner")


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / ".gitignore").write_text("ignored\n")
    (source / "tracked.py").write_text("original")
    (source / "ignored").write_text("do not upload")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    (source / "tracked.py").write_text("modified")
    (source / "untracked.py").write_text("untracked")
    return source


def submit(pool, source, code="print('ok')", timeout=10):
    driver = source.parent / ("driver-" + str(time.time_ns()) + ".py")
    driver.write_text(code)
    return pool.submit(source, driver, [], timeout, "test")


def test_snapshot_is_frozen_including_modified_and_untracked(source, tmp_path):
    pool = pool_module.Pool(tmp_path / "state")
    job = submit(pool, source)
    (source / "tracked.py").write_text("changed later")
    with tarfile.open(pool.jobdir(job) / "source.tar.gz") as bundle:
        assert bundle.extractfile("tracked.py").read() == b"modified"
        assert bundle.extractfile("untracked.py").read() == b"untracked"
        assert "ignored" not in bundle.getnames()
        assert not any(name.startswith(".git/") for name in bundle.getnames())


def test_concurrent_submission_and_claim_exactly_once(source, tmp_path):
    root = tmp_path / "state"
    pool_module.Pool(root)

    # Separate objects model separate agents, each using its own DB connection.
    def enqueue(_):
        return submit(pool_module.Pool(root), source)

    with ThreadPoolExecutor(max_workers=6) as executor:
        jobs = list(executor.map(enqueue, range(12)))
    code = (
        "import sys,json; sys.path.insert(0,sys.argv[1]); from pool import Pool; "
        "p=Pool(sys.argv[2]); out=[]; "
        "exec('while True:\\n j=p.claim(1)\\n if j is None: break\\n out.append(j)'); "
        "print(json.dumps(out))"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(SCRIPTS), str(root)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    claimed = []
    for process in processes:
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 0
        claimed.extend(json.loads(stdout))
    assert sorted(claimed) == sorted(jobs)


def test_wait_timeout_and_running_cancellation_do_not_release_job(source, tmp_path):
    pool = pool_module.Pool(tmp_path / "state")
    job = submit(pool, source)
    pool.claim(1)
    with pytest.raises(TimeoutError):
        pool.wait(job, 0)
    assert pool.get(job)["status"] == "running"
    assert pool_module.main(["--state-root", str(pool.root), "cancel", job]) == 1
    assert pool.get(job)["status"] == "running"


def test_supervisor_lock_survives_other_process(source, tmp_path):
    pool = pool_module.Pool(tmp_path / "state")
    with pool_module.exclusive(pool.root / "supervisor.lock"):
        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "pool.py"),
                "--state-root",
                str(pool.root),
                "recover",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert process.returncode == 1
        assert "already running" in process.stderr


class FakePool(pool_module.Pool):
    def __init__(self, root, *, fail=None, wrong_identity=False):
        super().__init__(root)
        self.fail = fail
        self.wrong_identity = wrong_identity
        self.events = []
        self.active = set()
        self.guard = threading.Lock()

    def call(self, slot, *args, timeout=180, log=None):
        if args[0] == "whoami":
            return (
                "Email: "
                + (
                    "other@example.com"
                    if self.wrong_identity
                    else pool_module.EXPECTED_EMAIL
                )
                + "\n"
            )
        if args[0] == "sessions":
            return (
                "\n".join(self.active)
                if self.active
                else "No active sessions found on server."
            )
        if args[0] == "new":
            name = args[2]
            self.active.add("endpoint-" + name)
            (self.slotdir(slot) / "sessions.json").write_text(
                json.dumps({name: {"name": name, "endpoint": "endpoint-" + name}})
            )
        if args[0] == "stop":
            if self.fail == "stop":
                raise RuntimeError("stop failed")
            self.active.discard("endpoint-" + args[2])
        return ""

    def execute(self, slot, job, session):
        with self.guard:
            assert not any(
                event[0] == "start" and event[1] == slot for event in self.events[-1:]
            )
            self.events.append(("start", slot, job, time.monotonic()))
        time.sleep(0.08)
        if self.fail == "execute":
            raise TimeoutError("connection lost with remote GPU possibly active")
        self.update(job, "succeeded")
        with self.guard:
            self.events.append(("end", slot, job, time.monotonic()))


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_worker_count_and_exclusive_sessions(source, tmp_path, workers):
    pool = FakePool(tmp_path / "state")
    jobs = [submit(pool, source) for _ in range(6)]
    pool.serve(workers, 0)
    assert all(pool.get(job)["status"] == "succeeded" for job in jobs)
    occupied = set()
    peak = 0
    for kind, slot, _, _ in pool.events:
        if kind == "start":
            assert slot not in occupied
            occupied.add(slot)
            peak = max(peak, len(occupied))
        else:
            occupied.remove(slot)
    assert peak == workers
    assert not pool.active
    assert all(pool.slotstate(i)["status"] == "stopped" for i in range(1, 4))


def test_execution_uncertainty_blocks_reuse_until_recovery(source, tmp_path):
    pool = FakePool(tmp_path / "state", fail="execute")
    first, second = [submit(pool, source) for _ in range(2)]
    with pytest.raises(RuntimeError, match="connection lost"):
        pool.serve(1, 0)
    assert pool.get(first)["status"] == "interrupted"
    assert pool.get(second)["status"] == "queued"
    assert not pool.active  # Stop was confirmed before releasing ownership.
    with pytest.raises(RuntimeError, match="recover"):
        pool.serve(1, 0)
    pool.recover()
    assert pool.get(first)["status"] == "failed"
    pool.fail = None
    pool.events.clear()  # Recovery confirmed the previous process is no longer running.
    pool.serve(1, 0)
    assert pool.get(second)["status"] == "succeeded"


def test_stop_failure_keeps_slot_quarantined(source, tmp_path):
    pool = FakePool(tmp_path / "state", fail="stop")
    submit(pool, source)
    with pytest.raises(RuntimeError, match="stop failed"):
        pool.serve(1, 0)
    assert pool.slotstate(1)["status"] == "quarantined"
    assert pool.active
    with pytest.raises(RuntimeError, match="recovery"):
        pool.serve(1, 0)
    pool.fail = None
    pool.recover()
    assert not pool.active


def test_wrong_identity_never_allocates(source, tmp_path):
    pool = FakePool(tmp_path / "state", wrong_identity=True)
    job = submit(pool, source)
    with pytest.raises(RuntimeError, match="identity"):
        pool.serve(1, 0)
    assert pool.get(job)["status"] == "queued"
    assert not pool.active


@pytest.mark.parametrize("kind", ["escape", "symlink", "duplicate"])
def test_unsafe_results_rejected(tmp_path, kind):
    directory = tmp_path / "job"
    directory.mkdir()
    with tarfile.open(directory / "results.tar.gz", "w:gz") as bundle:
        member = tarfile.TarInfo("../escape" if kind == "escape" else "file")
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
        bundle.addfile(member, io.BytesIO(b""))
        if kind == "duplicate":
            bundle.addfile(member, io.BytesIO(b""))
    with pytest.raises(ValueError, match="Unsafe"):
        pool_module.Pool.verify_results(directory, {})


@pytest.mark.parametrize(
    "code,timeout,expected",
    [
        (
            (
                "import os; from pathlib import Path; "
                "Path(os.environ['CST_JOB_OUTPUT'],'answer.txt').write_text('ok')"
            ),
            10,
            0,
        ),
        ("raise RuntimeError('driver failure')", 10, 1),
        ("import time; time.sleep(10)", 0.1, -9),
    ],
)
def test_real_child_process_success_failure_timeout(
    source, tmp_path, monkeypatch, code, timeout, expected
):
    pool = pool_module.Pool(tmp_path / "state")
    job = submit(pool, source, code, timeout)
    directory = pool.jobdir(job)
    spec = json.loads((directory / "spec.json").read_text())
    spec["remote_archive"] = str(directory / "source.tar.gz")
    spec["remote_result"] = str(directory / "results.tar.gz")
    # Fake only the device probe. Snapshot extraction, process execution,
    # timeout/kill, result archive and checksum verification are real.
    monkeypatch.setattr(
        remote.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "NVIDIA L4, UUID, driver, 23034 MiB"
            if command[0] == "nvidia-smi"
            else '{"torch":"test","cuda":"test","triton":"test"}'
        ),
    )
    remote.main(spec, tmp_path / "remote")
    pool.verify_results(directory, spec)
    result = json.loads((directory / "results/result.json").read_text())
    assert result["returncode"] == expected
    assert result["timed_out"] == (expected == -9)
    if expected == 0:
        assert (directory / "results/artifacts/answer.txt").read_text() == "ok"


def test_snapshot_symlink_rejected(source, tmp_path):
    (source / "link").symlink_to(source.parent / "driver.py")
    pool = pool_module.Pool(tmp_path / "state")
    with pytest.raises(ValueError, match="Symlinks"):
        submit(pool, source)
    assert not list((pool.root / "jobs").iterdir())


@pytest.mark.parametrize("hardware", ["NVIDIA A100", "NVIDIA RTX PRO 6000", "Tesla T4"])
def test_wrong_hardware_rejected_before_driver(source, tmp_path, monkeypatch, hardware):
    pool = pool_module.Pool(tmp_path / "state")
    job = submit(pool, source)
    spec = json.loads((pool.jobdir(job) / "spec.json").read_text())
    spec["remote_archive"] = str(pool.jobdir(job) / "source.tar.gz")
    monkeypatch.setattr(
        remote.subprocess,
        "check_output",
        lambda *a, **k: hardware + ", uuid, driver, memory",
    )
    with pytest.raises(ValueError, match="exactly one NVIDIA L4"):
        remote.main(spec, tmp_path / "remote")


def test_remaining_child_is_killed_after_normal_driver_exit(
    source, tmp_path, monkeypatch
):
    marker = tmp_path / "should-not-appear"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('alive')"
    driver = (
        f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    )
    pool = pool_module.Pool(tmp_path / "state")
    job = submit(pool, source, driver)
    directory = pool.jobdir(job)
    spec = json.loads((directory / "spec.json").read_text())
    spec["remote_archive"] = str(directory / "source.tar.gz")
    spec["remote_result"] = str(directory / "results.tar.gz")
    monkeypatch.setattr(
        remote.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "NVIDIA L4, UUID, driver, memory" if command[0] == "nvidia-smi" else "{}"
        ),
    )
    remote.main(spec, tmp_path / "remote")
    time.sleep(1.2)
    assert not marker.exists()


def test_interface_setting_is_scoped_to_child_environment(tmp_path, monkeypatch):
    pool = pool_module.Pool(tmp_path / "state", interface="en0")
    observed = {}
    previous = dict(os.environ)

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            observed.update(kwargs["env"])

        def communicate(self, **kwargs):
            return "done", None

    monkeypatch.setattr(pool_module.subprocess, "Popen", Process)
    assert pool.call(1, "sessions") == "done"
    assert observed["COLAB_POOL_INTERFACE"] == "en0"
    assert observed["PYTHONPATH"].startswith(str(SCRIPTS / "network"))
    assert dict(os.environ) == previous


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS socket binding")
def test_invalid_interface_fails_closed_without_running_cli(tmp_path):
    driver = tmp_path / "fake-colab"
    driver.write_text(f"#!{sys.executable}\nprint('CLI SHOULD NOT START')\n")
    driver.chmod(0o755)
    pool = pool_module.Pool(
        tmp_path / "state", cli=driver, interface="missing-interface-xyz"
    )
    log = tmp_path / "log"
    with pytest.raises(RuntimeError, match="failed"):
        pool.call(1, "sessions", log=log)
    assert "network setup failed" in log.read_text()
    assert "CLI SHOULD NOT START" not in log.read_text()


def test_idle_slot_stops_gpu_then_accepts_later_work_while_peer_runs(source, tmp_path):
    """A stopped GPU is not charged for idling, and its configured worker revives."""
    slow_started = threading.Event()
    release = threading.Event()

    class DelayedPool(FakePool):
        def execute(self, slot, job, session):
            if job != slow:
                return super().execute(slot, job, session)
            with self.guard:
                self.events.append(("start", slot, job, time.monotonic()))
            slow_started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the long experiment")
            self.update(job, "succeeded")
            with self.guard:
                self.events.append(("end", slot, job, time.monotonic()))

    def eventually(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    pool = DelayedPool(tmp_path / "state")
    slow = submit(pool, source)
    short = submit(pool, source)
    with ThreadPoolExecutor(max_workers=1) as executor:
        serving = executor.submit(pool.serve, 2, 0.02)
        try:
            assert slow_started.wait(3)
            eventually(lambda: pool.get(short)["status"] == "succeeded")
            slow_slot = pool.get(slow)["slot"]
            spare = 3 - slow_slot
            eventually(lambda: pool.slotstate(spare)["status"] == "stopped")
            assert pool.get(slow)["status"] == "running"
            assert len(pool.active) == 1
            later = submit(pool, source)
            eventually(lambda: pool.get(later)["status"] == "succeeded")
            assert pool.get(later)["slot"] == spare
            assert pool.get(slow)["status"] == "running"
            assert all(event[1] in (1, 2) for event in pool.events)
        finally:
            release.set()
        serving.result(timeout=3)
    assert not pool.active
    assert all(pool.slotstate(i)["status"] == "stopped" for i in (1, 2, 3))
