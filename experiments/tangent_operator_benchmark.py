"""Measure preparation, exact transport, and damped recompression separately.

The full weighted update solver is outside this benchmark. No accuracy claim.
CUDA peaks are incremental allocated bytes above each operation's live baseline;
prepared factors and dense reference matrices are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from torchcst import Chart, CSTLinear
from torchcst._derivatives.tangent import TangentGeometry
from torchcst._derivatives.tangent_solve import CompressionError, solve_compression
from torchcst.kernels import Amplitude, Gaussian, Separable
from torchcst.optim.moments import SeparableDiagonalMetric


def measure(fn, device, repeats):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    fn()
    sync()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        samples.append(1000 * (time.perf_counter() - start))
        del value
    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        value = fn()
        sync()
        peak = torch.cuda.max_memory_allocated(device) - base
        del value
    return {
        "median_ms": statistics.median(samples),
        "incremental_cuda_peak_bytes": peak,
    }


@torch.no_grad()
def run(k, inputs, outputs, args):
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(17)
    site = CSTLinear(
        Chart.linspace(inputs),
        Chart.linspace(outputs),
        atoms=k,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.25), output_profile=Gaussian(0.3))
        ),
        dtype=dtype,
        device=device,
    )
    p = site.atoms.p.detach().clone()
    old = p + 0.015 * torch.randn_like(p)
    x = torch.randn_like(p)
    ops = site.cst_derivatives().tangent_ops(atom_tile=args.tile)
    alternate = site.cst_derivatives().tangent_ops(
        backend="factor_autograd", atom_tile=args.tile
    )
    current, previous = ops.prepare(p), ops.prepare(old)
    geometry = TangentGeometry(ops.derivatives, ops=ops)
    # Match already-prepared factors for both contraction approaches.
    geometry._prepared = [(p, p._version, current), (old, old._version, previous)]
    geometry._parts(p)
    metric = SeparableDiagonalMetric(
        torch.rand(outputs, device=device, dtype=dtype),
        torch.rand(inputs, device=device, dtype=dtype),
        eps=1e-8,
    )
    rhs = current.gram_matvec(x)
    frame = geometry.frame(p)
    table = {}
    operations = {
        "prepare_specialized": lambda: ops.prepare(p),
        "prepare_factor_autograd": lambda: alternate.prepare(p),
        "transport_action": lambda: current.cross_gram_matvec(previous, x),
        "transport_full_gram": lambda: (
            geometry.cross(p, old) @ x.flatten()
        ).reshape_as(p),
        "weighted_action": lambda: current.weighted_gram_matvec(metric, x),
        "weighted_full_gram": lambda: geometry.cross(p, p, metric=metric) @ x.flatten(),
        "recompression_direct": lambda: geometry.compress(
            frame=frame, pullback_numerator=rhs, damping=args.damping
        ),
        "recompression_pcg": lambda: solve_compression(
            current, rhs, damping=args.damping, max_iter=args.max_iter, rtol=args.rtol
        ),
    }
    for name, fn in operations.items():
        try:
            table[name] = measure(fn, device, args.repeats)
        except CompressionError as error:
            table[name] = {"failure": asdict(error.result)}
    direct = operations["recompression_direct"]()
    direct_residual = float(
        (rhs - current.gram_matvec(direct) - args.damping * direct).double().norm()
        / rhs.double().norm().clamp_min(1e-30)
    )
    diagnostics = None
    try:
        alpha, diagnostics = operations["recompression_pcg"]()
        alpha_error = float(
            (alpha - direct).double().norm() / direct.double().norm().clamp_min(1e-30)
        )
        visible_direct = current.jvp(direct)
        visible_error = float(
            (current.jvp(alpha) - visible_direct).double().norm()
            / visible_direct.double().norm().clamp_min(1e-30)
        )
    except CompressionError as error:
        diagnostics = error.result
        alpha_error = None
        visible_error = None
    expected = operations["transport_full_gram"]()
    error = float(
        (operations["transport_action"]() - expected).double().norm()
        / expected.double().norm().clamp_min(1e-30)
    )
    factor_bytes = sum(
        t.numel() * t.element_size()
        for t in (current._u, current._v, current._du, current._dv, current._point)
    )
    return {
        "atoms": k,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": p.numel(),
        "single_prepared_factor_bytes": factor_bytes,
        "single_dense_gram_bytes": p.numel() ** 2 * p.element_size(),
        "transport_relative_error": error,
        "alpha_relative_error_vs_direct": alpha_error,
        "visible_history_relative_error_vs_direct": visible_error,
        "direct_true_relative_residual": direct_residual,
        "pcg": asdict(diagnostics),
        "timings": table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--atoms", nargs="+", type=int, default=[32, 85, 256])
    parser.add_argument("--inputs", type=int, default=784)
    parser.add_argument("--outputs", type=int, default=10)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    fingerprint = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        fingerprint.update(str(path.relative_to(root)).encode())
        fingerprint.update(path.read_bytes())
    result = {
        "environment": {
            "source_sha256": fingerprint.hexdigest(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name()
            if args.device.startswith("cuda")
            else None,
            "threads": 1,
        },
        "options": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
    }
    for atoms in args.atoms:
        case = run(atoms, args.inputs, args.outputs, args)
        result["cases"].append(case)
        print(json.dumps(case, allow_nan=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
