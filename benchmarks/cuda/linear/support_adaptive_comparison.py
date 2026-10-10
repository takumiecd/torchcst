"""Fixed Sphere sharp/main study using the unchanged full-step comparison worker.

The generic worker CLI keeps its original sigma3/8 interface. This benchmark
wrapper declares sigma1.25/3 and requires explicit Plans for CST comparisons.
"""

import argparse
import json
from pathlib import Path

from .manifest import REGISTRY
from .scaling_comparison import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--sigma", type=float, choices=(1.25, 3), required=True)
    parser.add_argument(
        "--mode", choices=("research_graph", "dense_graph"), required=True
    )
    parser.add_argument("--linear-plan", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "research_graph" and args.linear_plan is None:
        parser.error("research CST comparison requires an explicit Linear Plan")
    if args.mode == "dense_graph" and (args.linear_plan is not None or args.phases):
        parser.error("dense control has no CST Plan or phase diagnostics")
    plan = (
        None
        if args.linear_plan is None
        else REGISTRY.loads_plan(args.linear_plan.read_text())
    )
    result = worker(
        "sphere",
        args.size,
        args.sigma,
        args.mode,
        plan=plan,
        verify_only=args.verify_only,
        phases=args.phases,
    )
    result["study_condition"] = (
        "sharp diagnostic" if args.sigma == 1.25 else "ordinary sigma3 main"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "size",
                    "sigma_initial",
                    "study_condition",
                    "timing",
                    "peak_capture_replay",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
