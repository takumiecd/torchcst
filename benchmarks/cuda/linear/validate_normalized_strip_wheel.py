"""Validate the wheel in isolation, then benchmark its exported public class."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

out = Path(os.environ["CST_JOB_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
repo = Path.cwd()
wheels = out / "wheels"
wheels.mkdir(exist_ok=True)
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
(wheel,) = list(wheels.glob("torchcst-*.whl"))
package = out / "installed"
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(package)
assert not (package / "experiments").exists()
env = os.environ.copy()
env["CST_PUBLIC_PACKAGE_ROOT"] = str(package)
env["PYTHONPATH"] = os.pathsep.join([str(package), str(repo)])
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "pythonpath=" + str(package) + " .",
        "tests/test_normalized_strip_public.py",
        "tests/test_compact_profile.py",
    ],
    check=True,
    env=env,
)
driver = str(repo / "benchmarks/cuda/linear/check_normalized_strip_public.py")
small = out / "small_gate.json"
subprocess.run([sys.executable, driver, "--output", str(small)], check=True, env=env)
combined = json.loads(small.read_text())
combined["process_isolation"] = (
    "one fresh Python subprocess per complete-step case; stream/vendor workspaces from prior cases excluded"
)
combined["benchmarks"] = []
for n in (1024, 8192):
    cases = [
        (profile, memory, False)
        for profile in ("broad", "sharp")
        for memory in ("full", "window")
    ]
    cases.append(("broad", "full", True))
    for profile, memory, dense in cases:
        result = out / f"case-{n}-{profile}-{memory}-{int(dense)}.json"
        args = [
            sys.executable,
            driver,
            "--output",
            str(result),
            "--skip-small",
            "--bench",
            "--sizes",
            str(n),
            "--profiles",
            profile,
            "--memory",
            memory,
        ]
        if dense:
            args.append("--dense-only")
        subprocess.run(args, check=True, env=env)
        combined["benchmarks"].extend(json.loads(result.read_text())["benchmarks"])
        (out / "public_integration.json").write_text(
            json.dumps(combined, indent=2) + "\n"
        )
(out / "wheel_validation.json").write_text(
    json.dumps(
        {
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "package_root": str(package),
            "source_is_research_free": True,
        },
        indent=2,
    )
    + "\n"
)
