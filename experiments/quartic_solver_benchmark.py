"""Compare quartic solvers by setup-inclusive time and reference stationarity."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from torchcst import (
    AmplitudeBandwidthSeparable,
    BallNewton,
    Chart,
    CSTLinear,
    FullQuartic,
    Gaussian,
    ProjectedLBFGS,
    SubspaceQuartic,
)
from torchcst.optim import (
    AcceptedFrameFirstMoment,
    ImplicitLinearAtomGrad,
    MomentContext,
    MomentSystem,
    QuarticProblem,
    SeparableDiagonalSecondMoment,
)


class CountedProblem(QuarticProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = {
            name: 0 for name in ("value", "gradient", "value_and_gradient", "hessian")
        }

    def value(self, d):
        self.counts["value"] += 1
        return super().value(d)

    def gradient(self, d):
        self.counts["gradient"] += 1
        return super().gradient(d)

    def value_and_gradient(self, d):
        self.counts["value_and_gradient"] += 1
        return super().value_and_gradient(d)

    def hessian(self, d):
        self.counts["hessian"] += 1
        return super().hessian(d)

    def restricted_model(self, basis):
        self.counts["restricted_models"] = self.counts.get("restricted_models", 0) + 1
        return super().restricted_model(basis)


def fixture(seed, args):
    torch.manual_seed(seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    site = CSTLinear(
        Chart.grid((args.grid, args.grid)),
        Chart.linspace(10),
        atoms=args.atoms,
        kernel=AmplitudeBandwidthSeparable(
            input_profile=Gaussian(0.25), output_profile=Gaussian(0.1)
        ),
        dtype=dtype,
    ).to(device)
    system = MomentSystem(
        first=AcceptedFrameFirstMoment(0.9), second=SeparableDiagonalSecondMoment(0.99)
    )
    capture = ImplicitLinearAtomGrad(mode="custom", request=system.observation_request)
    site.atoms.set_grad(capture)
    inputs = torch.randn(16, site.in_features, device=device, dtype=dtype)
    targets = torch.randn(16, site.out_features, device=device, dtype=dtype)
    capture.begin()
    (site(inputs) - targets).square().mean().backward()
    capture.complete()
    geometry = site.cst_frame_geometry()
    context = MomentContext(geometry, geometry.current_point())
    moments = system.expand(system.initialize(context), capture.snapshot(), context)
    return site, moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--grid", type=int, default=28)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", default=["full", "projected", "newton10", "newton30"]
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    solvers = {
        "subspace16": SubspaceQuartic(
            max_models=24, max_dimension=16, tolerance_grad=1e-5
        ),
        "newton4": BallNewton(
            starts=4, max_iter=30, max_evaluations=150, tolerance_grad=1e-5
        ),
        "subspace": SubspaceQuartic(max_models=8, max_dimension=8, tolerance_grad=1e-5),
        "full": FullQuartic(starts=4, max_iter=80, tolerance_grad=1e-5),
        "projected": ProjectedLBFGS(
            max_evaluations=24, tolerance_grad=1e-5, relative_tolerance_grad=0
        ),
        "newton10": BallNewton(max_iter=10, max_evaluations=100, tolerance_grad=1e-5),
        "newton30": BallNewton(max_iter=30, max_evaluations=150, tolerance_grad=1e-5),
    }
    report = {
        "configuration": vars(args),
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "results": [],
    }
    for seed in args.seeds:
        site, moments = fixture(seed, args)
        observations = {name: [] for name in args.methods}
        for repetition in range(args.repeats + 1):
            for name in (
                args.methods if repetition % 2 == 0 else list(reversed(args.methods))
            ):
                synchronize()
                start = time.perf_counter()
                geometry = site.cst_frame_geometry()
                context = MomentContext(geometry, geometry.current_point())
                problem = CountedProblem(
                    context, moments, learning_rate=0.05, evaluation="visible"
                )
                result = solvers[name].solve(problem, trust_radius=args.radius)
                synchronize()
                duration = time.perf_counter() - start
                counts = problem.counts.copy()
                value, gradient = problem.value_and_gradient(result.displacement)
                residual = BallNewton._projected_norm(
                    result.displacement, gradient, args.radius
                )
                observation = {
                    "seconds": duration,
                    "objective": float(value),
                    "projected_gradient_norm": float(residual),
                    "norm": float(torch.linalg.vector_norm(result.displacement)),
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "counts": counts,
                }
                if repetition:
                    observations[name].append(observation)
        for name, rows in observations.items():
            record = {
                "seed": seed,
                "method": name,
                "median_seconds": statistics.median(row["seconds"] for row in rows),
                "samples": rows,
            }
            report["results"].append(record)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "method": name,
                        "seconds": record["median_seconds"],
                        "objective": rows[-1]["objective"],
                        "residual": rows[-1]["projected_gradient_norm"],
                        "counts": rows[-1]["counts"],
                    }
                ),
                flush=True,
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
