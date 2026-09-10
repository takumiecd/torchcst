"""Setup-inclusive comparison of exact visible and factored Gram quartics.

Run: PYTHONPATH=src .venv/bin/python -m experiments.quartic_gram_benchmark
The fixed problem comes from a synthetic forward/backward observation using
the four-coordinate MNIST kernel. No data downloads or training are required.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from torchcst import (
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    FullQuartic,
)
from torchcst.optim import (
    AcceptedFrameFirstMoment,
    ImplicitLinearAtomGrad,
    MomentContext,
    MomentSystem,
    QuarticProblem,
    SeparableDiagonalSecondMoment,
)


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    site = CSTLinear(
        Chart.grid((args.grid, args.grid)),
        Chart.linspace(args.outputs),
        atoms=args.atoms,
        kernel=AmplitudeBandwidthSeparable(sigma_min=0.1, sigma_max=1.0),
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
    expanded = system.expand(system.initialize(context), capture.snapshot(), context)
    displacement = torch.randn_like(context.current_point)
    displacement *= args.radius / torch.linalg.vector_norm(displacement)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def timed(function):
        synchronize()
        start = time.perf_counter()
        result = function()
        synchronize()
        return result, time.perf_counter() - start

    solver = FullQuartic(starts=args.starts, max_iter=args.iterations)
    records = {"visible": [], "gram": []}
    # One untimed warmup per backend; alternate the subsequent timing order.
    for repetition in range(args.repeats + 1):
        order = ("visible", "gram") if repetition % 2 == 0 else ("gram", "visible")
        for backend in order:
            # A fresh geometry prevents the visible derivative cache from
            # leaking between samples. First evaluation includes cache setup.
            fresh_context = MomentContext(
                site.cst_frame_geometry(), context.current_point
            )
            problem, setup = timed(
                lambda fresh_context=fresh_context, backend=backend: QuarticProblem(
                    fresh_context, expanded, learning_rate=0.05, evaluation=backend
                )
            )
            (value, gradient), first_eval = timed(
                lambda problem=problem: problem.value_and_gradient(displacement)
            )

            def evaluate_many(problem=problem):
                for _ in range(args.evaluations):
                    problem.value_and_gradient(displacement)

            _, evaluation_time = timed(evaluate_many)
            result, solve_time = timed(
                lambda problem=problem: solver.solve(problem, trust_radius=args.radius)
            )
            if repetition:
                records[backend].append(
                    {
                        "setup_seconds": setup + first_eval,
                        "value_gradient_seconds": evaluation_time / args.evaluations,
                        "setup_plus_evaluations_seconds": setup
                        + first_eval
                        + evaluation_time,
                        "solve_seconds": solve_time,
                        "setup_plus_solve_seconds": setup + first_eval + solve_time,
                    }
                )
            if backend == "visible":
                reference_value, reference_gradient = value, gradient
                reference_problem = problem
                reference_result = result
            else:
                gram_value, gram_gradient = value, gradient
                gram_result = result
    summary = {
        backend: {
            key: statistics.median(sample[key] for sample in samples)
            for key in samples[0]
        }
        for backend, samples in records.items()
    }
    summary["comparison"] = {
        "value_absolute_error": float((gram_value - reference_value).abs()),
        "gradient_relative_error": float(
            torch.linalg.vector_norm(gram_gradient - reference_gradient)
            / torch.linalg.vector_norm(reference_gradient).clamp_min(1e-30)
        ),
        "visible_solution_objective": float(reference_result.objective),
        "gram_solution_visible_objective": float(
            reference_problem.value(gram_result.displacement)
        ),
        "visible_projected_gradient": float(reference_result.projected_gradient_norm),
        "gram_projected_gradient": float(
            torch.linalg.vector_norm(
                gram_result.displacement
                - solver._project_ball(
                    gram_result.displacement
                    - reference_problem.gradient(gram_result.displacement),
                    args.radius,
                )
            )
        ),
        "evaluation_speedup": summary["visible"]["value_gradient_seconds"]
        / summary["gram"]["value_gradient_seconds"],
        "setup_plus_solve_speedup": summary["visible"]["setup_plus_solve_seconds"]
        / summary["gram"]["setup_plus_solve_seconds"],
    }
    return {
        "configuration": vars(args),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "cuda_version": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "results": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--grid", type=int, default=28)
    parser.add_argument("--outputs", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluations", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
