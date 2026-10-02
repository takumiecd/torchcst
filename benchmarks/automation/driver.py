"""Pool-only bootstrap: isolated environment, fixed snapshot, complete-step runner."""

import os
import subprocess
import sys
from pathlib import Path

from benchmarks.automation.contract import (
    encode,
    file_hash,
    validate_request,
    verify_source,
)


def run(request, job_key):
    validate_request(request)
    root = Path.cwd()
    verify_source(root, request)
    body = request["request"]
    job = next(j for j in body["jobs"] if j["key"] == job_key)
    out = Path(os.environ["CST_JOB_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_bytes(encode(request))
    snapshot = out / "run.json"
    snapshot.write_bytes(encode(job["run"]))
    venv = root / ".measurement-venv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(venv),
        ],
        check=True,
    )
    python = str(venv / "bin/python")
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.11.0+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    # Execute this module under the pinned environment before running any kernels.
    subprocess.run(
        [
            python,
            "-m",
            "benchmarks.automation.driver",
            str(out / "request.json"),
            job_key,
        ],
        check=True,
    )


def measure(request, job_key, out):
    import torch
    import triton

    from tools.colab.profiles import validate_hardware

    body = request["request"]
    prop = torch.cuda.get_device_properties(0)
    actual = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": prop.name,
        "compute_capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
    }
    (out / "environment.json").write_bytes(encode(actual))
    job = next(j for j in body["jobs"] if j["key"] == job_key)
    validate_hardware(job["accelerator"], actual["gpu"], actual["compute_capability"])
    if {k: actual[k] for k in body["environment"]} != body["environment"]:
        raise ValueError("GPU/runtime differs from the measurement request")
    snapshot = out / "run.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.run",
            "--snapshot",
            str(snapshot),
            "--snapshot-sha256",
            file_hash(snapshot),
            "--source-commit",
            body["source_commit"],
            "--output",
            str(out / "benchmark.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    from benchmarks.automation.contract import load_request

    path = Path(sys.argv[1])
    measure(load_request(path), sys.argv[2], path.parent)
