"""Compare matrix-free PCG against direct solves on real update frames.

One method/case per process. Reference matrices are constructed only after
measurement, so their storage cannot contaminate PCG peak memory.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.scaling_benchmark import model_for
from torchcst import CSTSecondOrderAdam, DeviceRay, SecondOrderAdamConfig
from torchcst._derivatives import factored_taylor as ft
from torchcst._derivatives._pcg import PCGOptions, solve
from torchcst._derivatives.factored_frame import FactoredFrameGeometry
from torchcst._derivatives.frame import GramSystem
from torchcst._runtime.validation import device_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["pcg", "cholesky"], required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    model = model_for("cst_ray1", args.width, args.width, args.atoms)
    optimizer = CSTSecondOrderAdam(
        model,
        cst=SecondOrderAdamConfig(
            lr=0.05,
            quartic=DeviceRay(corrections=1),
            device_execution=True,
            factored_geometry=True,
            gram_solver="cholesky",
            first_moment_damping=args.damping,
        ),
        dense=None,
    )

    class FrameCaptured(Exception):
        pass

    saved = {}

    def capture(self, *, frame, pullback_numerator, **kwargs):
        saved.update(
            f=tuple(
                t.detach().clone() for t in self.factor_local_derivatives(frame.point)
            ),
            d=frame.displacement.detach().clone(),
            b=pullback_numerator.detach().clone(),
        )
        raise FrameCaptured

    x = torch.randn(128, args.width, device="cuda")
    y = torch.randint(args.width, (128,), device="cuda")
    # Use an actual first-step rhs/frame, with all methods starting identically.
    with patch.object(FactoredFrameGeometry, "compress", capture):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        try:
            optimizer.step()
        except FrameCaptured:
            pass
    f, d, b = saved["f"], saved["d"], saved["b"]
    options = PCGOptions(args.iterations, args.rtol, args.block_size)

    @torch.no_grad()
    def run():
        if args.method == "pcg":
            return solve(f, d, b, damping=args.damping, options=options)
        matrix = ft.call("_gram_flat", *f, d)[0]
        with device_checks() as checks:
            result = GramSystem(
                matrix, d.shape, damping=args.damping, device_solver="cholesky"
            ).solve(b)
        return result, torch.stack(checks).all(), b.new_zeros(()), b.new_zeros(())

    run()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        torch.cuda.set_sync_debug_mode("error")
        start = time.perf_counter()
        result, valid, _residual, count = run()
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    with torch.no_grad():
        matrix64 = ft.frame_gram(tuple(t.double() for t in f), d.double())
        matrix64 += args.damping * torch.eye(
            d.numel(), device=d.device, dtype=torch.float64
        )
        oracle = torch.linalg.solve(matrix64, b.double().flatten())
        matrix32 = ft.frame_gram(f, d) + args.damping * torch.eye(
            d.numel(), device=d.device
        )
        direct32 = torch.linalg.solve(matrix32.double(), b.double().flatten())
        actual = result.double().flatten()
        error = (actual - oracle).norm() / oracle.norm().clamp_min(1e-30)
        direct_error = (actual - direct32).norm() / direct32.norm().clamp_min(1e-30)
        actual_residual = (matrix64 @ actual - b.flatten()).norm() / b.norm().clamp_min(
            1e-30
        )
    output = vars(args).copy()
    output["output"] = str(args.output)
    output.update(
        seconds=times,
        median_ms=1000 * statistics.median(times),
        baseline_allocated_mib=baseline / 2**20,
        peak_allocated_mib=peak / 2**20,
        incremental_peak_mib=(peak - baseline) / 2**20,
        peak_reserved_mib=reserved / 2**20,
        valid=bool(valid),
        relative_residual=float(actual_residual),
        relative_error_fp64_oracle=float(error),
        relative_error_existing_cholesky=float(direct_error),
        active_iterations=int(count),
        sync_debug_passed=True,
        note="isolated compression timing; setup stops before Gram construction; baseline retains proposal graphs",
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
