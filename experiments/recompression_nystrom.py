"""Setup-inclusive low-rank preconditioning experiment; not an optimizer default.

Replays the same captured full-operator PCG with fresh factors, sketch and RHS.
Dense matrices appear only in the independent oracle, outside measured regions.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from torchcst import Chart, CSTAdam, CSTLinear
from torchcst._derivatives.tangent_device import pcg
from torchcst._derivatives.tangent_nystrom import build
from torchcst._derivatives.tangent_ops import PreparedFactors
from torchcst._runtime.graphs import CapturedCall
from torchcst.kernels import Amplitude, Gaussian, Separable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--memory-mb", type=float, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    site = CSTLinear(
        Chart.linspace(784),
        Chart.linspace(10),
        atoms=args.atoms,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.25), output_profile=Gaussian(0.3))
        ),
        device="cuda",
        dtype=torch.float32,
    )
    # Match the first recompression RHS of the scaling experiment.
    optimizer = CSTAdam(
        site,
        device_execution=True,
        recompression="pcg",
        first_moment_damping=0.01,
        recompression_action="jvp_vjp",
        recompression_max_iter=1024,
        update_solver="krylov",
    )
    x = torch.randn(32, 784, device="cuda")
    target = torch.randn(32, 10, device="cuda")
    site_state = site.atoms.p.detach().clone()
    optimizer.zero_grad()
    (site(x) - target).square().mean().backward()
    # First EMA numerator is (1-beta1) times the pullback gradient.
    rhs = 0.1 * site.atoms.p.grad.detach()
    p = (
        site.cst_derivatives()
        .tangent_ops(execution="triton", gram_action="jvp_vjp")
        .prepare(site_state)
    )
    n = rhs.numel()
    rank = min(n, args.rank)
    if rank < 0 or 32 * n * max(rank, 1) > args.memory_mb * 2**20:
        raise ValueError("sketch budget exceeded (four FP64 N-by-r arrays)")
    generator = torch.Generator(device="cuda").manual_seed(831)
    omega = torch.linalg.qr(
        torch.randn(
            n, max(rank, 1), device="cuda", dtype=torch.float64, generator=generator
        )
    ).Q

    def run(point, u, v, du, dv, b, sketch):
        prepared = PreparedFactors(
            SimpleNamespace(
                backend="specialized", execution="triton", gram_action="jvp_vjp"
            ),
            point,
            (u, v, du, dv),
        )
        preconditioner = build(prepared, sketch, 0.01) if rank else None
        return pcg(
            prepared,
            b,
            damping=0.01,
            max_iter=1024,
            rtol=1e-5,
            compiled=True,
            nystrom=preconditioner,
        )

    call = CapturedCall(run)
    inputs = (p._point, p._u, p._v, p._du, p._dv, rhs, omega)
    call(*inputs)  # Compilation/capture warmup, includes setup in every graph replay.
    rows = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        alpha, count, relative, valid = call(*inputs)
        end.record()
        end.synchronize()
        rows.append(
            {
                "ms": start.elapsed_time(end),
                "iterations": count.item(),
                "relative_residual": relative.item(),
                "converged": valid.item(),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
        )
    # Independent FP64 residual through explicitly materialized Jacobian.
    j = (
        torch.einsum("ko,kiq->oikq", p._v.double(), p._du.double())
        + torch.einsum("koq,ki->oikq", p._dv.double(), p._u.double())
    ).reshape(-1, n)
    residual = (
        j.T @ (j @ alpha.double().flatten())
        + 0.01 * alpha.double().flatten()
        - rhs.double().flatten()
    )
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    report = {
        "atoms": args.atoms,
        "rank": rank,
        "sketch_budget_bytes": 32 * n * max(rank, 1),
        "rows": rows,
        "dense_relative_residual": (residual.norm() / rhs.double().norm()).item(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_sha256": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
