"""Shared L4 driver: installed CSTLinear, parameter reuse, Graph and step peaks.

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
        "tests/test_linear_parameters.py",
        "tests/test_atoms.py",
        "tests/test_normalized_strip_public.py",
        "tests/test_cuda_dispatch.py",
        "tests/test_operators.py",
        "tests/test_kernel_declarations.py",
        "tests/test_cst_optimizer.py",
        "tests/test_single_chart.py",
        "tests/test_triton_linear.py",
        "tests/test_coordinate_state.py",
    ],
    check=True,
    env=env,
)
public = str(repo / "benchmarks/cuda/linear/check_normalized_strip_public.py")
subprocess.run(
    [python, public, "--output", str(out / "graph-correctness.json")],
    check=True,
    env=env,
)
for profile in ("broad", "sharp"):
    for memory in ("full", "window"):
        subprocess.run(
            [
                python,
                public,
                "--bench",
                "--skip-small",
                "--sizes",
                "1024",
                "--profiles",
                profile,
                "--memory",
                memory,
                "--output",
                str(out / f"cst-{profile}-{memory}.json"),
            ],
            check=True,
            env=env,
        )
    # Dense baseline under the same batch, precision and optimizer contract.
    subprocess.run(
        [
            python,
            public,
            "--bench",
            "--skip-small",
            "--sizes",
            "1024",
            "--dense-only",
            "--output",
            str(out / f"dense-{profile}.json"),
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
            "scope": "installed wheel; selected CPU/CUDA contracts, independent mixed oracle, complete Graph training steps; no independent full-size all-atom gradient oracle",
            "process_memory_usage": "unmeasured",
        },
        indent=2,
    )
    + "\n"
)
