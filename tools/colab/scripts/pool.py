#!/usr/bin/env python3
"""Host-wide experiment queue for explicitly selected Colab GPU types.

Only the Python standard library and the installed colab CLI are needed.
Do not run direct colab commands against sessions owned by this pool.
"""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".local/state/colab-l4-pool"
DEFAULT_CLI = Path(shutil.which("colab") or Path.home() / ".local/bin/colab")
TERMINAL = {"succeeded", "failed", "cancelled"}

# Absolute invocation from a checkout or the legacy installed skill works alike.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.colab.profiles import ACCELERATORS  # noqa: E402


def atomic_json(path, data):
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "A pool supervisor is already running; submit to its queue"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Pool:
    def __init__(self, root=DEFAULT_ROOT, cli=DEFAULT_CLI, interface=None):
        self.root = Path(root).expanduser().resolve()
        self.cli = str(Path(cli).expanduser().resolve())
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings = self.root / "settings.json"
        config = json.loads(settings.read_text()) if settings.exists() else {}
        self.expected_account = config.get("account")
        self.accelerator = "L4"
        self.interface = interface or (config.get("interface"))
        (self.root / "jobs").mkdir(exist_ok=True)
        (self.root / "slots").mkdir(exist_ok=True)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, slot INTEGER, error TEXT,
                accelerator TEXT NOT NULL DEFAULT 'L4')""")
            if "accelerator" not in {
                row["name"] for row in db.execute("PRAGMA table_info(jobs)")
            }:
                db.execute(
                    "ALTER TABLE jobs ADD COLUMN accelerator TEXT NOT NULL DEFAULT 'L4'"
                )

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.root / "queue.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def jobdir(self, job):
        if not re.fullmatch(r"(?:l4job|colabjob)-[0-9a-f]{32}", job):
            raise ValueError("Invalid job ID")
        return self.root / "jobs" / job

    def get(self, job):
        self.jobdir(job)
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise ValueError("Unknown job ID")
        return dict(row)

    def update(self, job, status, error=None):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET status=?,updated=?,error=? WHERE id=?",
                (status, time.time(), error, job),
            )

    def claim(self, slot):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM jobs WHERE status='queued' AND accelerator=? ORDER BY created,id LIMIT 1",
                (self.accelerator,),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='running',slot=?,updated=? WHERE id=?",
                (slot, time.time(), row["id"]),
            )
            return row["id"]

    def submit(self, source, script, args, timeout, label, accelerator="L4"):
        if accelerator not in ACCELERATORS:
            raise ValueError("Unsupported Colab accelerator")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive")
        source, script = Path(source).resolve(), Path(script).resolve()
        if not source.is_dir() or not script.is_file():
            raise ValueError("Source directory and driver script must exist")
        job = ("l4job-" if accelerator == "L4" else "colabjob-") + uuid.uuid4().hex
        directory = self.jobdir(job)
        directory.mkdir()
        try:
            # Snapshot modified tracked files AND nonignored untracked files.
            # Git's ignore rules exclude venvs, credentials, output and .git.
            tracked = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(source),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                stderr=subprocess.PIPE,
            )
            top = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True
            ).strip()
            if Path(top).resolve() != source:
                raise ValueError("--source must be the Git working tree root")
            names = sorted({os.fsdecode(n) for n in tracked.split(b"\0") if n})
            hashes = {}
            with tarfile.open(directory / "source.tar.gz", "w:gz") as bundle:
                for name in names:
                    path = source / name
                    if path.is_symlink():
                        raise ValueError(
                            f"Symlinks are unsupported in source snapshots: {name}"
                        )
                    if not path.exists():
                        continue  # Deleted tracked files.
                    if not path.is_file() or not path.resolve().is_relative_to(source):
                        raise ValueError(
                            f"Unsupported source entry (including submodules): {name}"
                        )
                    if name == "__pool_driver__.py":
                        raise ValueError("Reserved source filename: __pool_driver__.py")
                    self.add_bytes(
                        bundle, name, path.read_bytes(), path.stat().st_mode, hashes
                    )
                self.add_bytes(
                    bundle, "__pool_driver__.py", script.read_bytes(), 0o644, hashes
                )
            archive_hash = file_hash(directory / "source.tar.gz")
            spec = {
                "id": job,
                "label": label,
                "source": str(source),
                "script": str(script),
                "args": args,
                "timeout": timeout,
                "accelerator": accelerator,
                "gpu_name_pattern": ACCELERATORS[accelerator][0],
                "source_sha256": archive_hash,
                "source_files": hashes,
                "submitted": time.time(),
                "remote_archive": f"/content/{job}.tar.gz",
                "remote_result": f"/content/{job}-results.tar.gz",
            }
            atomic_json(directory / "spec.json", spec)
            with self.db() as db:
                db.execute(
                    "INSERT INTO jobs (id,status,created,updated,accelerator) VALUES (?,?,?,?,?)",
                    (job, "queued", spec["submitted"], spec["submitted"], accelerator),
                )
        except BaseException:
            shutil.rmtree(directory)
            raise
        return job

    @staticmethod
    def add_bytes(bundle, name, data, mode, hashes):
        member = tarfile.TarInfo(name)
        member.size, member.mode = len(data), mode & 0o777
        bundle.addfile(member, io.BytesIO(data))
        hashes[name] = hashlib.sha256(data).hexdigest()

    def slotdir(self, slot):
        directory = self.root / "slots" / str(slot)
        directory.mkdir(exist_ok=True)
        return directory

    def slotstate(self, slot):
        path = self.slotdir(slot) / "state.json"
        return json.loads(path.read_text()) if path.exists() else {"status": "stopped"}

    def save_slot(self, slot, **data):
        atomic_json(self.slotdir(slot) / "state.json", data)

    def call(self, slot, *args, timeout=180, log=None):
        command = [
            self.cli,
            "--auth",
            "oauth2",
            "--config",
            str(self.slotdir(slot) / "sessions.json"),
            *map(str, args),
        ]
        # No shell interpolation. Kernel timeouts do not prove remote termination.
        environment = dict(os.environ)
        if self.interface:
            environment["COLAB_POOL_INTERFACE"] = self.interface
            environment["PYTHONPATH"] = (
                str(Path(__file__).with_name("network"))
                + os.pathsep
                + environment.get("PYTHONPATH", "")
            )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            if log:
                with Path(log).open("a") as handle:
                    handle.write(output)
            raise
        if log:
            with Path(log).open("a") as handle:
                handle.write(output)
        if process.returncode:
            raise RuntimeError(
                f"colab {args[0]} failed ({process.returncode}); inspect {log}"
            )
        return output

    def endpoint(self, slot, name):
        path = self.slotdir(slot) / "sessions.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())

        # Read only the endpoint, never copy tokens into manifests or logs.
        def find(value):
            if isinstance(value, dict):
                if value.get("name") == name and value.get("endpoint"):
                    return value["endpoint"]
                for child in value.values():
                    result = find(child)
                    if result:
                        return result
            elif isinstance(value, list):
                for child in value:
                    result = find(child)
                    if result:
                        return result
            return None

        return find(data)

    def start_slot(self, slot):
        name = "cst-pool-" + uuid.uuid4().hex[:12] + f"-{slot}"
        self.save_slot(slot, status="starting", session=name, endpoint=None)
        log = self.slotdir(slot) / "lifecycle.log"
        try:
            self.call(slot, "new", "-s", name, "--gpu", self.accelerator, log=log)
            endpoint = self.endpoint(slot, name)
            if not endpoint:
                raise RuntimeError("No endpoint persisted after allocation")
            self.save_slot(slot, status="ready", session=name, endpoint=endpoint)
        except BaseException:
            self.save_slot(
                slot,
                status="quarantined",
                session=name,
                endpoint=self.endpoint(slot, name),
            )
            raise
        return name

    def stop_slot(self, slot):
        state = self.slotstate(slot)
        if state["status"] == "stopped":
            return
        name = state["session"]
        endpoint = state.get("endpoint") or self.endpoint(slot, name)
        log = self.slotdir(slot) / "lifecycle.log"
        try:
            self.call(slot, "stop", "-s", name, log=log)
            listing = self.call(slot, "sessions", log=log)
            if not endpoint:
                # A failed allocation may have produced an untracked assignment.
                # Never destroy somebody else's VM to fix that ambiguity.
                if "No active sessions found on server." not in listing:
                    raise RuntimeError(
                        "Allocation endpoint unknown; inspect server assignments manually"
                    )
            elif endpoint in listing:
                raise RuntimeError("Server still lists the owned endpoint after stop")
            self.save_slot(slot, status="stopped", session=name, endpoint=endpoint)
        except BaseException:
            self.save_slot(slot, **{**state, "status": "quarantined"})
            raise

    def execute(self, slot, job, session):
        directory = self.jobdir(job)
        spec = json.loads((directory / "spec.json").read_text())
        log = directory / "transport.log"
        self.call(
            slot,
            "upload",
            "-s",
            session,
            directory / "source.tar.gz",
            spec["remote_archive"],
            log=log,
        )
        runner = Path(__file__).with_name("remote_runner.py").read_text()
        # Use an isolated namespace in the persistent notebook kernel.
        code = (
            "exec(compile("
            + repr(runner + "\nmain(" + repr(spec) + ")\n")
            + ', "pool_remote_runner", "exec"), {"__name__": "pool_remote_runner"})\n'
        )
        (directory / "remote.py").write_text(code)
        output = self.call(
            slot,
            "exec",
            "-s",
            session,
            "-f",
            directory / "remote.py",
            "--timeout",
            spec["timeout"] + 180,
            timeout=spec["timeout"] + 240,
            log=log,
        )
        if "POOL_RESULT_READY " + job not in output:
            raise RuntimeError(
                "Remote completion not acknowledged; runtime must be quarantined"
            )
        self.call(
            slot,
            "download",
            "-s",
            session,
            spec["remote_result"],
            directory / "results.tar.gz",
            log=log,
        )
        self.verify_results(directory, spec)
        result = json.loads((directory / "results/result.json").read_text())
        atomic_json(
            directory / "receipt.json",
            {
                "slot": slot,
                "session": session,
                "downloaded": time.time(),
                "archive_sha256": file_hash(directory / "results.tar.gz"),
            },
        )
        self.update(
            job,
            "succeeded"
            if result["returncode"] == 0 and not result["timed_out"]
            else "failed",
            None
            if result["returncode"] == 0 and not result["timed_out"]
            else f"Driver exit={result['returncode']}, timeout={result['timed_out']}",
        )
        # Remove remote source, artifacts and archives after verified download.
        cleanup = directory / "cleanup.py"
        cleanup.write_text(
            "import shutil\nfrom pathlib import Path\n"
            + f"shutil.rmtree({str('/content/' + job)!r})\n"
            + f"Path({spec['remote_archive']!r}).unlink()\n"
            + f"Path({spec['remote_result']!r}).unlink()\n"
            + f'print("POOL_CLEANED {job}")\n'
        )
        cleaned = self.call(
            slot, "exec", "-s", session, "-f", cleanup, "--timeout", 30, log=log
        )
        if f"POOL_CLEANED {job}" not in cleaned:
            raise RuntimeError("Remote cleanup not acknowledged")

    @staticmethod
    def verify_results(directory, spec):
        destination = directory / "results"
        destination.mkdir(exist_ok=False)
        with tarfile.open(directory / "results.tar.gz") as bundle:
            names = set()
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if (
                    not member.isfile()
                    or not target.is_relative_to(destination.resolve())
                    or member.name in names
                ):
                    raise ValueError("Unsafe or duplicate result archive member")
                names.add(member.name)
            bundle.extractall(destination)
        manifest = json.loads((destination / "manifest.json").read_text())
        actual = {
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file()
        }
        if actual != set(manifest) | {"manifest.json"}:
            raise ValueError(
                "Result manifest does not cover exactly the downloaded files"
            )
        for name, expected in manifest.items():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe result manifest path")
            if file_hash(path) != expected:
                raise ValueError(f"Result hash mismatch: {name}")
        result = json.loads((destination / "result.json").read_text())
        if (
            result["id"] != spec["id"]
            or result["source_sha256"] != spec["source_sha256"]
        ):
            raise ValueError("Result belongs to a different experiment")

    def worker(self, slot, idle_seconds, stopping):
        session = None
        idle_since = time.monotonic()
        try:
            while not stopping.is_set():
                job = self.claim(slot)
                if job is None:
                    if time.monotonic() - idle_since >= idle_seconds:
                        break
                    stopping.wait(0.25)
                    continue
                try:
                    state = self.slotstate(slot)
                    self.save_slot(slot, **{**state, "job": job})
                    session = session or self.start_slot(slot)
                    self.save_slot(slot, **{**self.slotstate(slot), "job": job})
                    self.execute(slot, job, session)
                    self.save_slot(slot, **{**self.slotstate(slot), "job": None})
                except BaseException as error:
                    # Never retry an experiment automatically; uncertain execution
                    # may already have changed its state or produced outputs.
                    if self.get(job)["status"] not in TERMINAL:
                        self.update(
                            job, "interrupted", f"{type(error).__name__}: {error}"
                        )
                    self.save_slot(
                        slot,
                        **{**self.slotstate(slot), "status": "quarantined", "job": job},
                    )
                    stopping.set()
                    raise
                idle_since = time.monotonic()
        finally:
            try:
                self.stop_slot(slot)
            except BaseException:
                # Idle cleanup uncertainty also stops all further dispatch.
                stopping.set()
                raise

    def recover(self):
        with exclusive(self.root / "supervisor.lock"):
            for slot in range(1, 4):
                self.stop_slot(slot)
            with self.db() as db:
                db.execute(
                    "UPDATE jobs SET status='failed',updated=?,error="
                    "'Previous execution interrupted; owned runtimes stopped. Resubmit explicitly.' "
                    "WHERE status IN ('running','interrupted')",
                    (time.time(),),
                )

    def serve(self, workers, idle_seconds, accelerator="L4"):
        if accelerator not in ACCELERATORS:
            raise ValueError("Unsupported Colab accelerator")
        self.accelerator = accelerator
        if not 1 <= workers <= 3 or not math.isfinite(idle_seconds) or idle_seconds < 0:
            raise ValueError("Use 1–3 workers and a nonnegative idle timeout")
        with exclusive(self.root / "supervisor.lock"):
            for slot in range(1, 4):
                if self.slotstate(slot)["status"] != "stopped":
                    raise RuntimeError(
                        "An owned runtime needs recovery. Run recover before serve."
                    )
            with self.db() as db:
                if db.execute(
                    "SELECT 1 FROM jobs WHERE status IN ('running','interrupted')"
                ).fetchone():
                    raise RuntimeError("Interrupted jobs need recover before serve")
            # Identity verification is read-only; no GPU allocated until a job is claimed.
            if not self.expected_account:
                raise RuntimeError("Configure the expected Colab account before serve")
            identity = self.call(1, "whoami", log=self.root / "identity.log")
            if not re.search(
                r"Email:\s*" + re.escape(self.expected_account) + r"\s", identity
            ):
                raise RuntimeError("Colab identity differs from the authorized account")
            self.call(1, "sessions", log=self.root / "sessions-before.log")
            stopping = threading.Event()
            previous = {}
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous[signum] = signal.signal(signum, lambda *_: stopping.set())
            errors = []
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self.worker, slot, idle_seconds, stopping): slot
                        for slot in range(1, workers + 1)
                    }
                    while futures:
                        done, _ = wait(
                            futures, timeout=0.25, return_when=FIRST_COMPLETED
                        )
                        for future in done:
                            del futures[future]
                            try:
                                future.result()
                            except Exception as error:  # noqa: BLE001 - preserve cleanup failures
                                errors.append(str(error))
                                stopping.set()
                        if stopping.is_set():
                            continue
                        with self.db() as db:
                            queued = db.execute(
                                "SELECT 1 FROM jobs WHERE status='queued' AND accelerator=? LIMIT 1",
                                (self.accelerator,),
                            ).fetchone()
                        if queued:
                            # Idle workers already stopped their VMs. Reuse only
                            # these configured slots when later work arrives;
                            # never increase the user's fixed worker count.
                            occupied = set(futures.values())
                            for slot in range(1, workers + 1):
                                if slot not in occupied:
                                    futures[
                                        executor.submit(
                                            self.worker, slot, idle_seconds, stopping
                                        )
                                    ] = slot
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
                try:
                    self.call(1, "sessions", log=self.root / "sessions-after.log")
                except Exception as error:  # noqa: BLE001 - preserve worker failures alongside cleanup errors
                    errors.append(str(error))
            if errors:
                raise RuntimeError("; ".join(errors))

    def wait(self, job, timeout):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Wait timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        while True:
            row = self.get(job)
            if row["status"] in TERMINAL or row["status"] == "interrupted":
                return row
            if time.monotonic() >= deadline:
                # This is a client wait timeout, NOT cancellation or lock release.
                raise TimeoutError(
                    f"Wait timed out; job {job} is still {row['status']}"
                )
            time.sleep(0.25)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Shared host queue. Override only for isolated tests.",
    )
    parser.add_argument("--colab", type=Path, default=DEFAULT_CLI)
    parser.add_argument(
        "--interface", help="macOS network interface for pool CLI processes only"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser(
        "submit", help="Snapshot a Git tree and enqueue a Python driver"
    )
    submit.add_argument("--source", required=True, type=Path)
    submit.add_argument("--script", required=True, type=Path)
    submit.add_argument("--label", default="experiment")
    submit.add_argument("--gpu", choices=ACCELERATORS, default="L4")
    submit.add_argument("--timeout", type=float, default=600)
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--wait-timeout", type=float, default=3600)
    submit.add_argument("args", nargs=argparse.REMAINDER)
    serve = sub.add_parser(
        "serve", help="Process the queue, stop owned VMs when idle, then exit"
    )
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--idle-seconds", type=float, default=60)
    serve.add_argument("--gpu", choices=ACCELERATORS, default="L4")
    wait = sub.add_parser("wait")
    wait.add_argument("job")
    wait.add_argument("--timeout", type=float, default=3600)
    sub.add_parser("status")
    cancel = sub.add_parser(
        "cancel", help="Cancel a queued job; running jobs keep their lease"
    )
    cancel.add_argument("job")
    sub.add_parser(
        "recover", help="Stop only pool-owned VMs and resolve interrupted jobs"
    )
    configure = sub.add_parser(
        "configure", help="Save an optional macOS CLI network interface"
    )
    configure.add_argument(
        "--interface",
        dest="saved_interface",
        help="Interface name, or 'default' for system routing",
    )
    configure.add_argument(
        "--account", help="expected authenticated Colab account email"
    )
    args = parser.parse_args(argv)
    pool = Pool(args.state_root, args.colab, args.interface)
    try:
        if args.command == "configure":
            import socket

            if args.saved_interface is None and args.account is None:
                raise ValueError("configure needs --account or --interface")
            interface = (
                None if args.saved_interface == "default" else args.saved_interface
            )
            if interface:
                if sys.platform != "darwin":
                    raise ValueError("Interface binding is supported only on macOS")
                socket.if_nametoindex(interface)
            with exclusive(pool.root / "supervisor.lock"):
                settings = pool.root / "settings.json"
                config = json.loads(settings.read_text()) if settings.exists() else {}
                if args.account is not None:
                    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", args.account):
                        raise ValueError("invalid account email")
                    config["account"] = args.account
                if args.saved_interface is not None:
                    config["interface"] = interface
                atomic_json(settings, config)
        elif args.command == "submit":
            driver_args = args.args[1:] if args.args[:1] == ["--"] else args.args
            job = pool.submit(
                args.source,
                args.script,
                driver_args,
                args.timeout,
                args.label,
                args.gpu,
            )
            print(
                json.dumps(
                    {"id": job, "directory": str(pool.jobdir(job)), "status": "queued"}
                ),
                flush=True,
            )
            if args.wait:
                row = pool.wait(job, args.wait_timeout)
                print(json.dumps(row))
                return 0 if row["status"] == "succeeded" else 1
        elif args.command == "serve":
            pool.serve(args.workers, args.idle_seconds, args.gpu)
        elif args.command == "recover":
            pool.recover()
        elif args.command == "wait":
            row = pool.wait(args.job, args.timeout)
            print(json.dumps(dict(**row, directory=str(pool.jobdir(args.job)))))
            return 0 if row["status"] == "succeeded" else 1
        elif args.command == "cancel":
            pool.get(args.job)
            with pool.db() as db:
                changed = db.execute(
                    "UPDATE jobs SET status='cancelled',updated=? "
                    "WHERE id=? AND status='queued'",
                    (time.time(), args.job),
                ).rowcount
            if not changed:
                raise RuntimeError(
                    "Only queued jobs can be cancelled; running jobs retain GPU ownership"
                )
        elif args.command == "status":
            with pool.db() as db:
                rows = [
                    dict(row)
                    for row in db.execute(
                        "SELECT * FROM jobs ORDER BY created DESC LIMIT 30"
                    )
                ]
            print(
                json.dumps(
                    {
                        "root": str(pool.root),
                        "jobs": rows,
                        "slots": {str(i): pool.slotstate(i) for i in range(1, 4)},
                    },
                    indent=2,
                )
            )
        return 0
    except (RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
