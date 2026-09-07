"""Compare identical one-start Newton budgets with compiled/GPU spectral kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from experiments import mnist_current_api as runner
from experiments.mnist_solver_comparison import fresh_problem, ranges
from torchcst import BallNewton
from torchcst.optim.solvers import QuarticSolver


def make_solver(name):
    return BallNewton(
        starts=1,
        max_iter=30,
        max_evaluations=150,
        execution="eager" if name == "eager" else "compiled",
        secular_solver="device" if name == "device" else "host",
    )


class Compare(QuarticSolver):
    def __init__(self, methods, output, repeats, profile):
        self.methods, self.output, self.repeats, self.profile = (
            methods,
            output,
            repeats,
            profile,
        )
        self.done = False

    def solve(self, problem, *, trust_radius):
        if self.done:
            return make_solver("eager").solve(problem, trust_radius=trust_radius)
        self.done = True
        device = problem.context.current_point.device
        records = {name: [] for name in self.methods}
        cold = {}
        baseline = None
        for repeat in range(self.repeats + 1):
            names = self.methods if repeat % 2 == 0 else list(reversed(self.methods))
            for name in names:
                runner._synchronize(device)
                start = time.perf_counter()
                candidate = make_solver(name).solve(
                    fresh_problem(problem, "visible"), trust_radius=trust_radius
                )
                runner._synchronize(device)
                elapsed = time.perf_counter() - start
                value, gradient = problem.value_and_gradient(candidate.displacement)
                row = {
                    "seconds": elapsed,
                    "objective": float(value),
                    "residual": float(
                        BallNewton._projected_norm(
                            candidate.displacement, gradient, trust_radius
                        )
                    ),
                    "evaluations": candidate.evaluations,
                    "iterations": candidate.iterations,
                    "converged": candidate.converged,
                }
                if repeat:
                    records[name].append(row)
                else:
                    cold[name] = row
                if name == "eager":
                    baseline = candidate
                print(json.dumps({"method": name, "repeat": repeat, **row}), flush=True)
                (self.output / "fixed.json").write_text(
                    json.dumps({"cold": cold, "samples": records}, indent=2)
                )
        if self.profile:
            for name in self.methods:
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with ranges(), torch.profiler.profile(activities=activities) as prof:
                    make_solver(name).solve(
                        fresh_problem(problem, "visible"), trust_radius=trust_radius
                    )
                    runner._synchronize(device)
                (self.output / f"{name}_profile.txt").write_text(
                    prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=50
                    )
                )
        (self.output / "fixed-summary.json").write_text(
            json.dumps(
                {
                    name: {
                        "median_seconds": statistics.median(r["seconds"] for r in rows),
                        "cold_seconds": cold[name]["seconds"],
                        "samples": rows,
                    }
                    for name, rows in records.items()
                },
                indent=2,
            )
        )
        return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=runner._default_data_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("fixed", "train"), default="fixed")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("eager", "compiled", "device"),
        default=["eager", "compiled", "device"],
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.stage == "fixed" and "eager" not in args.methods:
        parser.error("--stage fixed requires eager in --methods as the reference")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = runner.load_mnist(args.data, train_size=8192, test_size=2000)
    base = runner.ExperimentConfig(
        solver="newton",
        solver_starts=1,
        solver_max_iter=30,
        solver_max_evaluations=150,
        quartic_evaluation="visible",
    )
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": str(device),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "threads": torch.get_num_threads(),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device),
            },
            indent=2,
        )
    )
    if args.stage == "fixed":
        comparator = Compare(args.methods, args.output, args.repeats, args.profile)
        original = runner.build_optimizer

        def build(model, config):
            optimizer = original(model, config)
            optimizer.cst_config = replace(optimizer.cst_config, quartic=comparator)
            return optimizer

        with patch.object(runner, "build_optimizer", build):
            runner.run_seed(
                replace(base, steps=1, seed=args.seeds[0]), data, device=device
            )
    else:
        # Separate initial process/compilation warmup from measured learning.
        warmup = {}
        for name in args.methods:
            config = replace(
                base,
                steps=1,
                solver_execution="eager" if name == "eager" else "compiled",
                solver_secular="device" if name == "device" else "host",
            )
            warmup[name] = runner.run_seed(config, data, device=device)[
                "elapsed_seconds"
            ]
        (args.output / "warmup.json").write_text(json.dumps(warmup, indent=2))
        for index, seed in enumerate(args.seeds):
            names = args.methods if index % 2 == 0 else list(reversed(args.methods))
            for name in names:
                config = replace(
                    base,
                    steps=args.steps,
                    seed=seed,
                    solver_execution="eager" if name == "eager" else "compiled",
                    solver_secular="device" if name == "device" else "host",
                )
                result = runner.run_seed(config, data, device=device)
                (args.output / f"{name}_seed{seed}.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(
                    json.dumps(
                        {
                            "method": name,
                            "seed": seed,
                            "final": result["final"],
                            "seconds": result["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
