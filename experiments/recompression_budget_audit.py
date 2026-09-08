"""Freeze real training systems, then audit PCG fixed-schedule costs.

Short budgets are diagnostic counterfactuals, never applied to training.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments import local_tangent_learning
from torchcst._derivatives import tangent_device
from torchcst._derivatives.tangent_ops import PreparedFactors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    snapshots = []
    original = tangent_device.solve
    calls = 0

    def record(prepared, rhs, **kwargs):
        nonlocal calls
        calls += 1
        if calls in (32, 128, 384):
            saved = PreparedFactors(
                SimpleNamespace(
                    backend="specialized", execution="triton", gram_action="pair"
                ),
                prepared._point.clone(),
                tuple(
                    v.clone()
                    for v in (prepared._u, prepared._v, prepared._du, prepared._dv)
                ),
            )
            snapshots.append((calls, saved, rhs.clone()))
        return original(prepared, rhs, **kwargs)

    tangent_device.solve = record
    sys.argv = [
        "local_tangent_learning",
        "--data",
        str(args.data),
        "--output",
        str(args.output.with_suffix(".training.json")),
        "--approximation",
        "full",
        "--steps",
        "384",
        "--eval-every",
        "8",
        "--seed",
        "17",
    ]
    local_tangent_learning.main()
    tangent_device.solve = original
    training = json.loads(args.output.with_suffix(".training.json").read_text())
    assert training["status"] == "passed"
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "training_source_sha256": training["source_sha256"],
        "snapshots": [],
    }

    for step, prepared, rhs in snapshots:

        def run(budget, value, prepared=prepared):
            return original(prepared, value, damping=0.01, max_iter=budget, rtol=1e-5)

        reference = run(1024, rhs)
        torch.cuda.synchronize()
        count = int(reference[1])
        assert bool(reference[3])
        results = []
        for name, budget, value in [
            ("full", 1024, rhs),
            ("cut_after_convergence", count + 1, rhs),
            ("half_budget", 512, rhs),
            ("zero_budget", 0, rhs),
            ("inactive_full", 1024, torch.zeros_like(rhs)),
            ("inactive_zero", 0, torch.zeros_like(rhs)),
        ]:
            run(budget, value)
            run(budget, value)
            torch.cuda.synchronize()
            times = []
            for _ in range(15):
                begin = time.perf_counter()
                out = run(budget, value)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - begin) * 1000)
            diff = (out[0] - reference[0]).norm() / reference[0].norm().clamp_min(1e-30)
            results.append(
                {
                    "name": name,
                    "budget": budget,
                    "median_ms": statistics.median(times),
                    "iterations": int(out[1]),
                    "relative_residual": float(out[2]),
                    "valid": bool(out[3]),
                    "solution_relative_difference": float(diff),
                    "times_ms": times,
                }
            )
            if name in ("full", "half_budget", "cut_after_convergence"):
                assert bool(out[3]) and float(diff) == 0
        report["snapshots"].append(
            {"step": step, "active_iterations": count, "results": results}
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["snapshots"][-1]), flush=True)


if __name__ == "__main__":
    main()
