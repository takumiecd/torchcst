"""L4 pool driver: installed-wheel contracts, independent correctness and step peaks."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

out = Path(os.environ["CST_JOB_OUTPUT"])
repo = Path.cwd()
wheels = out / "wheels"
wheels.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-deps",
        "--wheel-dir",
        str(wheels),
    ],
    check=True,
)
(wheel,) = list(wheels.glob("*.whl"))
installed = out / "installed"
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(installed)
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([str(installed), str(repo)])
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "pythonpath=" + str(installed) + " .",
        "tests/test_cuda_dispatch.py",
        "tests/test_normalized_strip_public.py",
        "tests/test_compact_profile.py",
    ],
    check=True,
    env=env,
)
subprocess.run(
    [
        sys.executable,
        str(repo / "benchmarks/cuda/linear/check_normalized_strip_public.py"),
        "--output",
        str(out / "graph_correctness.json"),
    ],
    check=True,
    env=env,
)
for profile in ("broad", "sharp"):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.run",
            "--algorithm",
            "normalized_window",
            "--size",
            "1024",
            "--profile",
            profile,
            "--dense",
            "--source-commit",
            sys.argv[1],
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
            "source_commit": sys.argv[1],
            "wheel": wheel.name,
            "cases": ["broad", "sharp"],
            "size": 1024,
            "scope": "initial dispatch contract; isolated complete steps including Graph peak; no performance promotion",
        },
        indent=2,
    )
    + "\n"
)
