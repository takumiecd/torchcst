"""Prepare comparisons, validate declarations and run kernel test suites.

This tool does not measure performance, submit jobs or change dispatch policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.cuda.linear.manifest import (
    DEFAULT_PLANS,
    REGISTRY,
    decode_catalog,
    decode_snapshot,
    load_run,
    read_json,
)

ROOT = Path(__file__).resolve().parents[2]
GPU_SUITES = {
    "local-product": (
        "tests/test_local_product_research.py",
        "tests/test_local_polar_update.py",
        "tests/test_local_persistent_layout.py",
        "tests/test_local_contraction.py",
        "tests/test_local_launch.py",
    ),
    "normalized-strip": (
        "tests/test_normalized_strip_public.py",
        "tests/test_cuda_dispatch.py",
        "tests/test_dispatch_selectors.py",
    ),
}


def check(plans: Path, cases: list[Path]):
    """Use the runner's contracts; do not maintain another manifest validator."""
    value, digest = read_json(plans)
    entries = decode_catalog(value)
    for entry in entries:
        if REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) != entry.plan:
            raise ValueError(f"plan does not round-trip: {entry.id}")
    results = []
    for path in cases:
        run = load_run(path, plans)
        # Exercise the same JSON representation used by isolated GPU workers.
        snapshot = json.loads(json.dumps(run.snapshot(), allow_nan=False))
        if decode_snapshot(snapshot) != run:
            raise ValueError(f"run snapshot does not round-trip: {path}")
        results.append(
            {
                "path": str(path),
                "id": run.case.id,
                "fixture": run.case.fixture,
                "plans": [entry.id for entry in run.plans],
                "baseline": run.baseline,
                "dense": run.dense,
                "input_hashes": run.input_hashes,
            }
        )
    return {
        "status": "PASS",
        "scope": "declarations_only",
        "catalog": str(plans),
        "catalog_sha256": digest,
        "plan_count": len(entries),
        "cases": results,
    }


def test_suite(suite: str):
    if suite == "cpu":
        import torch

        if torch.cuda.is_available():
            raise ValueError(
                "cpu suite requires CUDA to be unavailable; "
                "run with CUDA_VISIBLE_DEVICES='' on a GPU host"
            )
        paths = ("tests",)
    else:
        import torch

        if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
            raise ValueError(f"{suite} suite requires NVIDIA CUDA and Triton")
        paths = GPU_SUITES[suite]
    # Preserve pytest's exit status, including test failures and collection errors.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths], cwd=ROOT, check=False
    ).returncode


def prepare(plans: Path, case: Path, candidates: list[str], output: Path):
    """Create a small baseline/candidate comparison using registered plans."""
    source = load_run(case, plans)
    catalog_value, catalog_hash = read_json(plans)
    if catalog_hash != source.input_hashes["plans_sha256"]:
        raise ValueError("source catalog changed while preparing comparison")
    entries = decode_catalog(catalog_value)
    by_id = {entry.id: entry for entry in entries}
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or source.baseline in candidates
    ):
        raise ValueError("candidates must be unique and different from the baseline")
    if any(name not in by_id for name in candidates):
        raise ValueError("candidate is not registered in the source catalog")
    if not source.dense:
        raise ValueError("comparison source Case must include the dense reference")
    selected = [source.baseline, *candidates]
    # This also rejects candidates with a different mathematical contract.
    run = replace(source, plans=tuple(by_id[name] for name in selected))
    catalog = {
        "schema_version": 1,
        "plans": [
            {"id": entry.id, "plan": REGISTRY.dump_plan(entry.plan)}
            for entry in run.plans
        ],
    }
    declaration = {
        "schema_version": 1,
        "case": asdict(run.case),
        "plans": selected,
        "baseline": run.baseline,
        "dense": run.dense,
    }
    provenance = {
        "plans": str(plans),
        "case": str(case),
        "input_hashes": source.input_hashes,
    }
    # Validate everything before creating files; never overwrite an experiment.
    payloads = {
        name: json.dumps(value, indent=2, allow_nan=False) + "\n"
        for name, value in (
            ("plans.json", catalog),
            ("case.json", declaration),
            ("source.json", provenance),
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, text in payloads.items():
        (output / name).write_text(text)
    return check(output / "plans.json", [output / "case.json"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("check", help="CPU-only Plan/Case validation")
    preflight.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    preflight.add_argument("--case", type=Path, action="append", default=[])
    comparison = commands.add_parser(
        "prepare", help="create a baseline/candidate comparison from registered plans"
    )
    comparison.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    comparison.add_argument("--case", type=Path, required=True)
    comparison.add_argument("--candidate", action="append", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    tests = commands.add_parser("test", help="run a documented test suite")
    tests.add_argument("--suite", choices=("cpu", *GPU_SUITES), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(check(args.plans, args.case), indent=2, allow_nan=False))
            return 0
        if args.command == "prepare":
            print(
                json.dumps(
                    prepare(args.plans, args.case, args.candidate, args.output),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        return test_suite(args.suite)
    except (OSError, ValueError, TypeError) as error:
        print(f"kernel-dev: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
