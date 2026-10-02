"""Colab host adapter. Never operates owned pool sessions directly."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.automation.contract import (
    decode,
    encode,
    file_hash,
    keys,
    load_request,
    relative_path,
    sha,
    verify_source,
)


def pool_call(pool, *args, check=True):
    completed = subprocess.run(
        [sys.executable, str(pool), *map(str, args)], capture_output=True, text=True
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "shared pool command failed")
    return completed


def submit(request_file, root, output, pool):
    request = load_request(request_file)
    verify_source(root, request, checkout=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = base64.b64encode(encode(request)).decode("ascii")
    pending = {"schema_version": 1, "request_id": request["request_id"], "jobs": []}
    for job in request["request"]["jobs"]:
        driver = output / f"{job['key']}.py"
        driver.write_text(
            "import base64,json\nfrom benchmarks.automation.driver import run\n"
            f"run(json.loads(base64.b64decode({payload!r})), {job['key']!r})\n"
        )
        response = decode(
            pool_call(
                pool,
                "submit",
                "--source",
                root,
                "--script",
                driver,
                "--timeout",
                request["request"]["timeout_seconds"],
                "--label",
                f"benchmark-{job['key']}",
                "--gpu",
                job["accelerator"],
            ).stdout
        )
        pending["jobs"].append(
            {
                "key": job["key"],
                "pool_job_id": response["id"],
                "directory": response["directory"],
            }
        )
        (output / "pending.json").write_bytes(encode(pending))
    return pending


def collect(request_file, pending_file, output, pool):
    request = load_request(request_file)
    pending = decode(Path(pending_file).read_bytes())
    keys(pending, ("schema_version", "request_id", "jobs"), "pending batch")
    expected_keys = {j["key"] for j in request["request"]["jobs"]}
    if (
        pending["schema_version"] != 1
        or pending["request_id"] != request["request_id"]
        or len(pending["jobs"]) != len(expected_keys)
        or {j["key"] for j in pending["jobs"]} != expected_keys
    ):
        raise ValueError("pending batch differs from request")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = output / "records"
    records.mkdir()
    bundle = {"schema_version": 1, "request_id": request["request_id"], "jobs": []}
    (output / "submission.json").write_bytes(encode(bundle))
    for entry in pending["jobs"]:
        # Wait in bounded increments. A timeout never cancels or releases a VM.
        while True:
            response = pool_call(
                pool, "wait", entry["pool_job_id"], "--timeout", 30, check=False
            )
            if response.stdout.strip():
                state = decode(response.stdout)
                if state["status"] == "interrupted":
                    raise RuntimeError(
                        "pool job interrupted; inspect status and follow pool recovery before collecting"
                    )
                if state["status"] not in ("queued", "running", "interrupted"):
                    break
            elif (
                "Timed out" not in response.stderr
                and "timeout" not in response.stderr.lower()
            ):
                raise RuntimeError(
                    response.stderr.strip() or "cannot inspect shared pool job"
                )
        directory = Path(entry["directory"]).resolve()
        spec = decode((directory / "spec.json").read_bytes())
        if spec["id"] != entry["pool_job_id"]:
            raise ValueError("pool job identity mismatch")
        source = spec["source_files"]
        requested_job = next(
            j for j in request["request"]["jobs"] if j["key"] == entry["key"]
        )
        if spec.get("accelerator", "L4") != requested_job["accelerator"]:
            raise ValueError("pool job requested a different GPU")
        if {k: v for k, v in source.items() if k != "__pool_driver__.py"} != request[
            "request"
        ]["source_files"]:
            raise ValueError("pool snapshot differs from request")
        row = {
            "key": entry["key"],
            "pool_job_id": entry["pool_job_id"],
            "status": state["status"],
            "source_archive_sha256": spec["source_sha256"],
            "result_archive_sha256": None,
            "artifact": None,
            "artifact_sha256": None,
        }
        receipt_file = directory / "receipt.json"
        if receipt_file.exists():
            receipt = decode(receipt_file.read_bytes())
            if file_hash(directory / "results.tar.gz") != receipt["archive_sha256"]:
                raise ValueError("pool result archive differs from retrieval receipt")
            row["result_archive_sha256"] = receipt["archive_sha256"]
            result_dir = directory / "results"
            manifest = decode((result_dir / "manifest.json").read_bytes())
            actual_files = {
                str(p.relative_to(result_dir))
                for p in result_dir.rglob("*")
                if p.is_file()
            }
            if actual_files != set(manifest) | {"manifest.json"}:
                raise ValueError(
                    "retrieved manifest does not cover exactly the result files"
                )
            for name, expected in manifest.items():
                relative_path(name)
                sha(expected)
                path = result_dir / name
                if path.is_symlink() or file_hash(path) != expected:
                    raise ValueError("retrieved pool file hash mismatch")
            result = decode((result_dir / "result.json").read_bytes())
            if (
                result["id"] != entry["pool_job_id"]
                or result["source_sha256"] != spec["source_sha256"]
            ):
                raise ValueError("retrieved result belongs to a different source/job")
            artifact = result_dir / "artifacts/benchmark.json"
            if artifact.exists():
                row["artifact"] = f"records/{entry['key']}.json"
                shutil.copyfile(artifact, output / row["artifact"])
                row["artifact_sha256"] = file_hash(artifact)
            for name in ("stdout.log", "stderr.log"):
                if (result_dir / name).exists():
                    shutil.copyfile(
                        result_dir / name, output / f"{entry['key']}-{name}"
                    )
        bundle["jobs"].append(row)
        (output / "submission.json").write_bytes(encode(bundle))
    return bundle


def run(request_file, root, output, pool):
    """Submit, share/start one supervisor, then retrieve all verified results."""
    output = Path(output)
    submit(request_file, root, output / "pending", pool)
    for target in load_request(request_file)["request"]["targets"]:
        accelerator = target["accelerator"]
        pending = decode((output / "pending/pending.json").read_bytes())
        jobs = {
            j["key"]
            for j in load_request(request_file)["request"]["jobs"]
            if j["accelerator"] == accelerator
        }
        while True:
            waiting = []
            for entry in pending["jobs"]:
                if entry["key"] not in jobs:
                    continue
                state = pool_call(
                    pool, "wait", entry["pool_job_id"], "--timeout", 0, check=False
                )
                if not state.stdout.strip():
                    if (
                        "timeout" not in state.stderr.lower()
                        and "timed out" not in state.stderr.lower()
                    ):
                        raise RuntimeError(
                            state.stderr.strip() or "cannot inspect queued job"
                        )
                    waiting.append(entry["pool_job_id"])
                elif decode(state.stdout)["status"] == "interrupted":
                    raise RuntimeError(
                        "pool job interrupted; follow the pool recovery procedure"
                    )
            if not waiting:
                break
            served = pool_call(
                pool,
                "serve",
                "--workers",
                1,
                "--idle-seconds",
                0,
                "--gpu",
                accelerator,
                check=False,
            )
            with (output / "supervisor.log").open("a") as log:
                log.write(served.stdout + served.stderr)
            if served.returncode:
                if "A pool supervisor is already running" not in served.stderr:
                    raise RuntimeError(
                        "pool supervisor failed; inspect its log and status; no automatic recovery"
                    )
                # Do not overlap an existing owner or change its selected GPU.
                # Bounded wait, then retry only after its process releases the lock.
                pool_call(pool, "wait", waiting[0], "--timeout", 30, check=False)
    return collect(
        request_file, output / "pending/pending.json", output / "results", pool
    )
