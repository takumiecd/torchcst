"""Shape/atom scaling of complete warmed training updates on one CUDA device.

Run each case in a fresh process so captured graphs cannot contaminate memory
measurements. Synthetic batches measure runtime, not accuracy equivalence.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.mnist_current_api import ExperimentConfig
from torchcst import (
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    CSTOptimizer,
    DeviceRay,
    Gaussian,
    ImplicitAdamConfig,
)


def model_for(method, inputs, outputs, atoms):
    if method == "dense_adam":
        return torch.nn.Linear(inputs, outputs, bias=False).cuda()
    cfg = ExperimentConfig()
    side = math.isqrt(inputs)
    while inputs % side:
        side -= 1
    return CSTLinear(
        Chart.grid((side, inputs // side)),
        Chart.linspace(outputs),
        atoms=atoms,
        kernel=AmplitudeBandwidthSeparable(
            input_profile=Gaussian(cfg.input_sigma),
            output_profile=Gaussian(cfg.output_sigma),
            sigma_explore=cfg.sigma_explore,
            tau=cfg.tau,
            temperature=cfg.temperature,
        ),
        atom_init="uniform",
        backend="factored",
        dtype=torch.float32,
    ).cuda()


def measure(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    if args.derivative_cache_mib is not None:
        from torchcst._derivatives.frame import AutogradFrameGeometry

        # Benchmark-only override; the library default remains unchanged.
        AutogradFrameGeometry._MAX_DERIVATIVE_CACHE_ELEMENTS = int(
            args.derivative_cache_mib * 2**20 / 4
        )
    model = model_for(args.method, args.inputs, args.outputs, args.atoms)
    if args.method == "cst_ray1":
        cfg = ExperimentConfig()
        optimizer = CSTOptimizer(
            model,
            cst=ImplicitAdamConfig(
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.epsilon,
                trust_radius=cfg.trust_radius,
                quartic=DeviceRay(corrections=1),
                quartic_evaluation="visible",
                device_execution=True,
                factored_geometry=args.factored_geometry,
                gram_solver=args.gram_solver,
                gram_iterations=args.gram_iterations,
                gram_rtol=args.gram_rtol,
                gram_block_size=args.gram_block_size,
                first_moment_damping=args.damping,
            ),
            dense=None,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, betas=(0.9, 0.99), foreach=True
        )
    # Eight resident batches; all methods receive identical data for each shape.
    generator = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn(8, args.batch, args.inputs, device="cuda", generator=generator)
    y = torch.randint(args.outputs, (8, args.batch), device="cuda", generator=generator)

    def step(i):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[i % 8]), y[i % 8])
        loss.backward()
        optimizer.step()
        return loss

    started = time.perf_counter()
    for i in range(8):
        step(i)
    torch.cuda.synchronize()
    if args.method == "cst_ray1":
        optimizer.check_errors()
    cold_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    times = []
    torch.cuda.set_sync_debug_mode("error")
    try:
        for repeat in range(args.repeats):
            # Explicit timing boundary; no synchronization inside the step loop.
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            started = time.perf_counter()
            for i in range(args.steps):
                loss = step(i + repeat * args.steps)
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            torch.cuda.set_sync_debug_mode("error")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if args.method == "cst_ray1":
        optimizer.check_errors()
    return {
        "factored_geometry": args.factored_geometry,
        "gram_solver": args.gram_solver,
        "gram_iterations": args.gram_iterations,
        "gram_rtol": args.gram_rtol,
        "gram_block_size": args.gram_block_size,
        "first_moment_damping": args.damping,
        "method": args.method,
        "inputs": args.inputs,
        "outputs": args.outputs,
        "atoms": args.atoms if args.method != "dense_adam" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "derivative_cache_limit_mib": args.derivative_cache_mib,
        "batch": args.batch,
        "steps_per_repeat": args.steps,
        "seconds": times,
        "median_ms_per_step": statistics.median(times) * 1000 / args.steps,
        "cold_eight_steps_seconds": cold_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "final_loss": loss.item(),
        "sync_debug_passed": True,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["dense_adam", "cst_adam", "cst_ray1"], required=True
    )
    parser.add_argument("--inputs", type=int, required=True)
    parser.add_argument("--outputs", type=int, required=True)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--factored-geometry", action="store_true")
    parser.add_argument(
        "--gram-solver", choices=["jacobi", "cholesky", "pcg"], default="jacobi"
    )
    parser.add_argument("--gram-iterations", type=int, default=64)
    parser.add_argument("--gram-rtol", type=float, default=1e-5)
    parser.add_argument("--gram-block-size", type=int, default=32)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--derivative-cache-mib", type=float)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = measure(args)
    except (RuntimeError, FloatingPointError) as error:
        failure = {
            "status": "failed",
            "error": str(error),
            "method": args.method,
            "inputs": args.inputs,
            "outputs": args.outputs,
            "atoms": args.atoms,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        args.output.with_suffix(".failure.json").write_text(
            json.dumps(failure, indent=2)
        )
        raise
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
