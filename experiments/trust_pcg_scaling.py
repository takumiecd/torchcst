"""One isolated CUDA scaling trial; retain failed steps instead of timing no-ops.

Run each shape/solver in a fresh process so CUDA Graph caches cannot contaminate
the other solver's memory. Checks and diagnostic copies follow measured scopes.
The first two steps cover initial capture and the first transport of history;
both are excluded from warm medians.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

from torchcst import Chart, CSTAdam, CSTLinear
from torchcst.kernels import Amplitude, Gaussian, Separable


class TimedAdam(CSTAdam):
    def _solve(self, context, expanded):
        self.solve_start.record()
        result = super()._solve(context, expanded)
        self.solve_end.record()
        return result


def diagnostics(result):
    names = (
        "iterations",
        "evaluations",
        "converged",
        "on_boundary",
        "shift",
        "relative_residual",
        "relative_complementarity",
        "shift_iterations",
        "projected_gradient_norm",
        "objective",
    )
    return {
        name: value.item() if isinstance(value, torch.Tensor) else value
        for name in names
        if (value := getattr(result, name, None)) is not None
    }


def memory():
    return {
        name: getattr(torch.cuda, function)() / 2**20
        for name, function in (
            ("allocated_mib", "memory_allocated"),
            ("reserved_mib", "memory_reserved"),
            ("peak_allocated_mib", "max_memory_allocated"),
            ("peak_reserved_mib", "max_memory_reserved"),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--input", type=int, default=784)
    parser.add_argument("--output-width", type=int, default=10)
    parser.add_argument("--solver", choices=("spectral", "pcg"), required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--backend", choices=("auto", "factored", "materialized"), default="auto"
    )
    parser.add_argument("--update-max-iter", type=int, default=512)
    parser.add_argument("--update-shift-steps", type=int, default=32)
    parser.add_argument("--compression-max-iter", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py"))
    sources += sorted((root / "src").rglob("*.cpp")) + [Path(__file__).resolve()]
    report = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "rows": [],
        "status": "running",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        site = CSTLinear(
            Chart.linspace(args.input),
            Chart.linspace(args.output_width),
            atoms=args.atoms,
            backend=args.backend,
            kernel=Amplitude(
                Separable(input_profile=Gaussian(0.25), output_profile=Gaussian(0.3))
            ),
            dtype=torch.float32,
            device="cuda",
        )
        optimizer = TimedAdam(
            site,
            device_execution=True,
            update_solver=args.solver,
            recompression="pcg",
            first_moment_damping=0.01,
            recompression_max_iter=args.compression_max_iter,
            update_max_iter=args.update_max_iter,
            update_shift_steps=args.update_shift_steps,
            update_rtol=1e-5,
        )
        optimizer.solve_start = torch.cuda.Event(enable_timing=True)
        optimizer.solve_end = torch.cuda.Event(enable_timing=True)
        x = torch.randn(32, args.input, device="cuda")
        target = torch.randn(32, args.output_width, device="cuda")
        report["parameters"] = site.atoms.p.numel()
        report["resolved_backend"] = site._resolved_backend()
        report["baseline_memory"] = memory()
        for step in range(args.steps):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            optimizer.zero_grad()
            loss = (site(x) - target).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = {
                "step": step,
                "warmup": step < 2,
                "seconds": seconds,
                "update_ms": optimizer.solve_start.elapsed_time(optimizer.solve_end),
                "memory": memory(),
                "loss": loss.item(),
                "compression": diagnostics(optimizer.last_step.compression_results[0]),
                "update": diagnostics(optimizer.last_step.site_results[0]),
            }
            report["rows"].append(row)
            # A failed deferred step latches the optimizer. Never time another
            # iteration after it or treat suppressed state commits as throughput.
            save()
            optimizer.check_errors()
            if not row["update"]["converged"]:
                raise RuntimeError("update solver reported non-convergence")
            row["accepted"] = True
            save()
            print(json.dumps(row), flush=True)
        warm = report["rows"][2:]
        report["status"] = "passed"
        if warm:
            report["warm_median_ms"] = (
                statistics.median(r["seconds"] for r in warm) * 1000
            )
            report["warm_update_median_ms"] = statistics.median(
                r["update_ms"] for r in warm
            )
            report["warm_peak_allocated_mib"] = max(
                r["memory"]["peak_allocated_mib"] for r in warm
            )
            report["warm_peak_reserved_mib"] = max(
                r["memory"]["peak_reserved_mib"] for r in warm
            )
    except (RuntimeError, ValueError, AssertionError, ArithmeticError) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "source_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
