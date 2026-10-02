"""L4 wheel gate: all maintained tests, independent truth, Graph and step peaks.

Use a job-local CUDA 12.8 environment matching the preceding validated runtime.
No system package mutations; every performance case runs in a separate process.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_commit")
args = parser.parse_args()
repo = Path.cwd()
out = Path(os.environ["CST_JOB_OUTPUT"])
venv = repo / ".linear-validation-cu128"
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
runtime = json.loads(
    subprocess.check_output(
        [
            python,
            "-c",
            'import json,torch,triton; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"triton":triton.__version__,"gpu":torch.cuda.get_device_name()}))',
        ],
        text=True,
    )
)
assert runtime["cuda"] == "12.8" and runtime["gpu"] == "NVIDIA L4", runtime
(out / "validation-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
wheels = out / "wheels"
wheels.mkdir()
subprocess.run(
    [python, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheels)],
    check=True,
)
(wheel,) = wheels.glob("*.whl")
installed = out / "installed"
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(installed)
env = dict(
    os.environ,
    PYTHONPATH=os.pathsep.join([str(installed), str(repo)]),
    CST_PUBLIC_PACKAGE_ROOT=str(installed),
)
subprocess.run(
    [
        python,
        "-m",
        "pytest",
        "-q",
        "-o",
        "pythonpath=" + str(installed) + " .",
        "--junitxml=" + str(out / "pytest.xml"),
        "tests",
    ],
    check=True,
    env=env,
)
public = str(repo / "benchmarks/cuda/linear/check_normalized.py")
subprocess.run(
    [python, public, "--output", str(out / "graph-correctness.json")],
    check=True,
    env=env,
)
for profile in ("broad", "sharp"):
    # Resolve the catalog once; every correctness/timing worker consumes the
    # same hash-checked snapshot and forces the selected registry Plan.
    subprocess.run(
        [
            python,
            "-m",
            "benchmarks.cuda.linear.run",
            "--case",
            str(repo / f"benchmarks/cuda/linear/cases/normalized-1024-{profile}.json"),
            "--source-commit",
            args.source_commit,
            "--output",
            str(out / f"plans-{profile}.json"),
        ],
        check=True,
        env=env,
    )
(out / "summary.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "source_commit": args.source_commit,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "runtime": runtime,
            "scope": "installed wheel; all maintained CPU/CUDA contracts, independent mixed oracle, complete Graph training steps; no independent full-size all-atom gradient oracle",
            "process_memory_usage": "unmeasured",
        },
        indent=2,
    )
    + "\n"
)
