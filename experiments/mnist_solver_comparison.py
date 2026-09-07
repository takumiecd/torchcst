"""Controlled MNIST learning comparison and same-state solver profiling.

Profiling is a separate baseline trajectory: replayed solvers never commit
parameters or moments. Timings from instrumented runs are not speedup evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.mnist_current_api import (
    ExperimentConfig,
    _default_data_root,
    _synchronize,
    load_mnist,
    run_seed,
)
from torchcst import BallNewton, FullQuartic
from torchcst.optim import MomentContext, QuarticProblem
from torchcst.optim.solvers import QuarticSolver
from torchcst.optim.solvers import newton as newton_module


@contextmanager
def ranges():
    """Trace CPU dispatch and CUDA activity; nested ranges are inclusive."""

    def decorate(function, label):
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return function(*args, **kwargs)

        return wrapped

    with ExitStack() as stack:
        for owner, name, label in [
            *[
                (QuarticProblem, name, f"cst::{name}")
                for name in (
                    "__init__",
                    "value",
                    "gradient",
                    "value_and_gradient",
                    "hessian",
                )
            ],
            (torch.Tensor, "backward", "cst::autograd_backward"),
            (torch.linalg, "eigh", "cst::eigh"),
            (newton_module, "_spectral_ball_minimum", "cst::spectral_ball"),
        ]:
            stack.enter_context(
                patch.object(owner, name, decorate(getattr(owner, name), label))
            )
        yield


def fresh_problem(problem, evaluation):
    # A new frame geometry prevents derivative caches from favoring later methods.
    geometry = type(problem.context.geometry)(problem.context.geometry.derivatives)
    return QuarticProblem(
        MomentContext(geometry, problem.context.current_point.clone()),
        problem.moments,
        learning_rate=problem.learning_rate,
        evaluation=evaluation,
    )


class TimedSolver(QuarticSolver):
    """A separate diagnostic run with synchronized solver boundaries."""

    def __init__(self, solver):
        self.solver = solver
        self.seconds = []

    def solve(self, problem, *, trust_radius):
        device = problem.context.current_point.device
        _synchronize(device)
        started = time.perf_counter()
        result = self.solver.solve(problem, trust_radius=trust_radius)
        _synchronize(device)
        self.seconds.append(time.perf_counter() - started)
        return result


class ReplaySolver(QuarticSolver):
    def __init__(self, methods, checkpoints, repeats, directory):
        self.methods, self.checkpoints = methods, checkpoints
        self.repeats, self.directory = repeats, directory
        self.step = 0
        self.records = []

    def solve(self, problem, *, trust_radius):
        self.step += 1
        if self.step not in self.checkpoints:
            return FullQuartic(starts=4, max_iter=80).solve(
                problem, trust_radius=trust_radius
            )
        device = problem.context.current_point.device
        reference = fresh_problem(problem, "visible")
        samples = {name: [] for name in self.methods}
        baseline = None
        for repeat in range(self.repeats + 1):
            names = list(self.methods)
            if repeat % 2:
                names.reverse()
            for name in names:
                solver, evaluation = self.methods[name]
                _synchronize(device)
                start = time.perf_counter()
                candidate = solver.solve(
                    fresh_problem(problem, evaluation), trust_radius=trust_radius
                )
                _synchronize(device)
                seconds = time.perf_counter() - start
                value, gradient = reference.value_and_gradient(candidate.displacement)
                row = {
                    "seconds": seconds,
                    "objective": float(value),
                    "residual": float(
                        BallNewton._projected_norm(
                            candidate.displacement, gradient, trust_radius
                        )
                    ),
                    "converged": candidate.converged,
                }
                if repeat:
                    samples[name].append(row)
                if name == "full":
                    baseline = candidate
        for name, (solver, evaluation) in self.methods.items():
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with (
                ranges(),
                torch.profiler.profile(activities=activities) as profile,
                torch.profiler.record_function("cst::setup_and_solve"),
            ):
                solver.solve(
                    fresh_problem(problem, evaluation), trust_radius=trust_radius
                )
                _synchronize(device)
            stem = self.directory / f"step{self.step}_{name}"
            # Keep aggregate tables only: full traces can be hundreds of MB.
            events = profile.key_averages()
            stem.with_suffix(".txt").write_text(
                events.table(sort_by="self_cpu_time_total", row_limit=40)
            )
            record = {
                "step": self.step,
                "method": name,
                "samples": samples[name],
                "median_seconds": statistics.median(
                    r["seconds"] for r in samples[name]
                ),
                "ranges": [
                    {
                        "name": e.key,
                        "device_type": str(e.device_type),
                        "calls": e.count,
                        "cpu_total_ms": e.cpu_time_total / 1000,
                        "self_cpu_ms": e.self_cpu_time_total / 1000,
                        "device_total_ms": getattr(e, "device_time_total", 0) / 1000,
                    }
                    for e in events
                    if e.key.startswith("cst::")
                ],
            }
            self.records.append(record)
            print(
                json.dumps({k: v for k, v in record.items() if k != "ranges"}),
                flush=True,
            )
        (self.directory / "paired.json").write_text(json.dumps(self.records, indent=2))
        return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data", type=Path, default=_default_data_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument(
        "--evaluation", choices=("auto", "visible", "gram"), default="auto"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--phase-timing", action="store_true")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1, 4, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = load_mnist(args.data, train_size=8192, test_size=2000)
    base = ExperimentConfig(steps=args.steps, quartic_evaluation=args.evaluation)
    configurations = {
        "full": base,
        "newton1": replace(
            base,
            solver="newton",
            solver_starts=1,
            solver_max_iter=30,
            solver_max_evaluations=150,
        ),
        "newton4": replace(
            base,
            solver="newton",
            solver_starts=4,
            solver_max_iter=30,
            solver_max_evaluations=150,
        ),
    }
    environment = {
        "torch": torch.__version__,
        "device": str(device),
        "threads": 1,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    if args.profile:
        methods = {
            "full": (FullQuartic(starts=4, max_iter=80), args.evaluation),
            "newton1": (
                BallNewton(starts=1, max_iter=30, max_evaluations=150),
                args.evaluation,
            ),
            "newton4": (
                BallNewton(starts=4, max_iter=30, max_evaluations=150),
                args.evaluation,
            ),
        }
        replay = ReplaySolver(methods, set(args.checkpoints), args.repeats, args.output)
        from experiments import mnist_current_api as runner

        original = runner.build_optimizer

        def build(model, config):
            optimizer = original(model, config)
            optimizer.cst_config = replace(optimizer.cst_config, quartic=replay)
            return optimizer

        with patch.object(runner, "build_optimizer", build):
            run_seed(replace(base, seed=args.seeds[0]), dataset, device=device)
    else:
        # Warm up independently. Reset model and minibatch RNG for every real run.
        run_seed(replace(base, steps=1), dataset, device=device)
        for index, seed in enumerate(args.seeds):
            names = list(configurations)
            if index % 2:
                names.reverse()
            for name in names:
                config = replace(configurations[name], seed=seed)
                if args.phase_timing:
                    from experiments import mnist_current_api as runner

                    original = runner.build_optimizer
                    timers = []

                    def timed_build(model, config, original=original, timers=timers):
                        optimizer = original(model, config)
                        timer = TimedSolver(optimizer.cst_config.quartic)
                        timers.append(timer)
                        optimizer.cst_config = replace(
                            optimizer.cst_config, quartic=timer
                        )
                        return optimizer

                    with patch.object(runner, "build_optimizer", timed_build):
                        result = run_seed(config, dataset, device=device)
                    result["solver_seconds"] = timers[0].seconds
                    result["solver_fraction"] = sum(timers[0].seconds) / sum(
                        row["step_seconds"] for row in result["trace"]
                    )
                else:
                    result = run_seed(config, dataset, device=device)
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
