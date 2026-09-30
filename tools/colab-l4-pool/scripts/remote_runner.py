"""Run inside a Colab kernel without importing CUDA libraries into that kernel."""

import hashlib
import json
import os
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


def main(spec, content_root="/content"):
    base = Path(content_root) / spec["id"]
    source = base / "source"
    results = base / "results"
    artifacts = results / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    archive = Path(spec["remote_archive"])
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
    if not gpu.startswith("NVIDIA L4,") or len(gpu.splitlines()) != 1:
        raise ValueError(f"Expected exactly one NVIDIA L4, got {gpu!r}")
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
