"""Run inside a Colab kernel without importing CUDA libraries into that kernel."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.isfile():
                raise ValueError(
                    "Archive must contain regular files inside destination"
                )
        bundle.extractall(destination)


def terminate_group(process):
    # Also kill child processes left behind after the driver itself exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def owned_archive(spec, content_root):
    job = spec.get("id")
    if (
        not isinstance(job, str)
        or not re.fullmatch(r"(?:l4job|colabjob)-[0-9a-f]{32}", job)
        or spec.get("remote_archive") != str(Path(content_root) / f"{job}.tar.gz")
    ):
        raise ValueError("Source archive is not the exact job-owned path")
    return Path(spec["remote_archive"])


def assemble_source(spec, parts, content_root="/content"):
    """Verify exact owned chunks and the original archive before extraction."""
    archive = owned_archive(spec, content_root)
    if not isinstance(parts, list) or not parts:
        raise ValueError("Source chunks must be a nonempty list")
    for index, part in enumerate(parts):
        expected = f"{archive}.part-{index:05d}"
        if (
            not isinstance(part, dict)
            or set(part) != {"path", "bytes", "sha256"}
            or part["path"] != expected
            or type(part["bytes"]) is not int
            or not 0 < part["bytes"] <= 32 * 1024 * 1024
            or not isinstance(part["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])
            or Path(expected).is_symlink()
            or not Path(expected).is_file()
            or Path(expected).stat().st_size != part["bytes"]
        ):
            raise ValueError("Invalid source chunk descriptor or file")
    # Exclusive creation preserves any pre-existing archive. Failed assembly
    # retains chunks/partial archive for the supervisor's quarantine path.
    with archive.open("xb") as output:
        for part in parts:
            digest = hashlib.sha256()
            with Path(part["path"]).open("rb") as handle:
                for data in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(data)
                    output.write(data)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError("Uploaded source chunk SHA256 mismatch")
    if file_hash(archive) != spec["source_sha256"]:
        raise ValueError("Assembled source SHA256 mismatch")
    for part in parts:
        Path(part["path"]).unlink()


def main(spec, content_root="/content", source_parts=None):
    # Preserve historical single-upload callers; chunk assembly additionally
    # requires the exact owned archive under the runner's content root.
    if source_parts:
        owned_archive(spec, content_root)
    base = Path(content_root) / spec["id"]
    source = base / "source"
    results = base / "results"
    artifacts = results / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    archive = Path(spec["remote_archive"])
    if source_parts:
        assemble_source(spec, source_parts, content_root)
    if file_hash(archive) != spec["source_sha256"]:
        raise ValueError("Uploaded source SHA256 mismatch")
    unpack(archive, source)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=30,
    ).strip()
    expected = spec.get("gpu_name_pattern", r"NVIDIA L4")
    if len(gpu.splitlines()) != 1 or not re.fullmatch(
        expected, gpu.split(",")[0].strip()
    ):
        raise ValueError(f"Expected exactly one requested GPU, got {gpu!r}")
    versions = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,torch; import importlib.metadata as m; "
                "print(json.dumps(dict(python=sys.version,torch=torch.__version__,"
                'cuda=torch.version.cuda,triton=m.version("triton"))))'
            ),
        ],
        text=True,
        timeout=60,
    ).strip()
    env = dict(os.environ)
    env.update(
        CST_JOB_ID=spec["id"],
        CST_JOB_OUTPUT=str(artifacts),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(source / "src") + os.pathsep + str(source),
    )
    started = time.time()
    timed_out = False
    with (
        (results / "stdout.log").open("w") as stdout,
        (results / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, str(source / "__pool_driver__.py"), *spec["args"]],
            cwd=source,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.wait(timeout=spec["timeout"])
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            terminate_group(process)
    result = {
        "id": spec["id"],
        "source_sha256": spec["source_sha256"],
        "gpu": gpu,
        "requested_accelerator": spec.get("accelerator", "L4"),
        "versions": json.loads(versions),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "started": started,
        "finished": time.time(),
        "timeout_scope": "driver process, including any setup inside driver",
    }
    (results / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    files = [
        p
        for p in results.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.resolve().is_relative_to(results.resolve())
    ]
    manifest = {str(p.relative_to(results)): file_hash(p) for p in files}
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = Path(spec["remote_result"])
    with tarfile.open(output, "w:gz") as bundle:
        for p in [*files, results / "manifest.json"]:
            bundle.add(p, arcname=str(p.relative_to(results)), recursive=False)
    print("POOL_RESULT_READY " + spec["id"], flush=True)
