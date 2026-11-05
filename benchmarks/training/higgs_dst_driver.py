"""Shared-L4 driver for the exploratory, development-only DST pilot.

Run via pool --script with --source-commit FULL_SHA. This driver never allocates
resources or installs system packages. The source/data manifests are prepared
by the submitting orchestrator after committing this separate research tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def verify_inputs(source, commit, executed_driver):
    from benchmarks.training.higgs_confirmation_bootstrap import (
        file_hash,
        read_source_manifest,
    )

    manifest, inputs = read_source_manifest(source, commit)
    name = "benchmarks/training/higgs_dst_driver.py"
    if manifest["files"].get(name) != file_hash(executed_driver):
        raise ValueError("executed pool driver differs from frozen tracked source")
    return {
        "source_commit": commit,
        "source_manifest_sha256": inputs["source-manifest.json"],
        "dataset_sha256": inputs["screen.npz"],
        "data_manifest_sha256": inputs["data-manifest.json"],
        "verified_tracked_files": len(manifest["files"]),
    }


def probe_runtime(env):
    # Keep CUDA/Torch initialization in a fresh child, as for all pool scripts.
    code = """
import json, platform, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('pilot requires exactly one CUDA device')
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if name != 'NVIDIA L4' or capability != (8, 9):
    raise RuntimeError(f'pilot requires NVIDIA L4 CC8.9, got {name} {capability}')
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'gpu': name,
                  'capability': list(capability)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return json.loads(completed.stdout)


def write_record(output, record):
    (output / "dst-pilot-driver.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def run(args):
    sys.dont_write_bytecode = True
    source = Path.cwd().resolve()
    output = Path(os.environ["CST_JOB_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "stage": args.stage,
        "status": "running",
        "purpose": "exploratory learning dynamics; not tuned accuracy or efficiency evidence",
        "development_partition": [0, 50000],
        "confirmation_evaluated": False,
        "test_evaluated": False,
        "commands": [],
    }
    begin = time.perf_counter()
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        MAX_JOBS="2",
        TORCH_CUDA_ARCH_LIST="8.9",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    try:
        record["source"] = verify_inputs(source, args.source_commit, Path(__file__))
        record["runtime"] = probe_runtime(env)
        write_record(output, record)
        # Qualification is a runner-owned CPU oracle + GPU VJP/Graph gate.
        commands = [
            [
                sys.executable,
                "-m",
                "benchmarks.training.higgs_dst_reference",
                "--qualify-only",
                "--output",
                str(output / "qualification"),
            ]
        ]
        if args.stage == "pilot":
            for kind in ("static", "rigl"):
                commands.append(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.training.higgs_dst_reference",
                        "--data",
                        str(source / "research-inputs/screen.npz"),
                        "--output",
                        str(output / kind),
                        "--kind",
                        kind,
                        "--width",
                        "1024",
                        "--divisor",
                        "1000",
                        "--seed",
                        "17",
                        "--lr",
                        "0.0003",
                        "--epochs",
                        "10",
                        "--mode",
                        "graph",
                    ]
                )
        for command in commands:
            item = {"argv": command, "status": "running"}
            record["commands"].append(item)
            write_record(output, record)
            subprocess.run(command, check=True, cwd=source, env=env)
            item["status"] = "passed"
            write_record(output, record)
        record["status"] = "passed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - begin
        write_record(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("qualify", "pilot"), default="pilot")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
