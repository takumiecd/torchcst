"""Measure bounded-window Triton GEMMs with IEEE and TF32x3 accumulation."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton as tr
import triton.language as tl


@tr.jit
def gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSK: tl.constexpr,
    BSN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tr.cdiv(K, BK)):
        ks = offset * BK + inner
        a = tl.load(
            A + rows[:, None] * ASM + ks[None, :] * ASK,
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        b = tl.load(
            B + ks[:, None] * BSK + cols[None, :] * BSN,
            (ks[:, None] < K) & (cols[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def run_gemm(a, b, c, bm, bn, bk, precision, warps):
    gemm_kernel[(tr.cdiv(a.shape[0], bm), tr.cdiv(b.shape[1], bn))](
        a,
        b,
        c,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        *a.stride(),
        *b.stride(),
        bm,
        bn,
        bk,
        precision,
        num_warps=warps,
    )


def milliseconds(fn, repeats=5):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compare(actual, expected):
    difference = (actual - expected).abs()
    tolerance = 3e-4 + 3e-4 * expected.abs()
    return {
        "max_abs": difference.max().item(),
        "relative_l2": (difference.norm() / expected.norm()).item(),
        "violations_3e_4": (difference > tolerance).sum().item(),
        "elements": actual.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, 1024, device="cuda")
    w = torch.randn(1024, n, device="cuda")
    cases = ((dy.T, x, "dw"), (dy, w, "dx"))
    results = []
    for a, b, name in cases:
        reference = a @ b
        c = torch.empty_like(reference)
        for precision in ("ieee", "tf32x3", "tf32"):
            shapes = (
                (32, 64, 32, 4),
                (32, 128, 32, 4),
                (64, 64, 32, 4),
                (32, 64, 64, 4),
                (64, 128, 32, 8),
            )
            for bm, bn, bk, warps in shapes:
                run = partial(run_gemm, a, b, c, bm, bn, bk, precision, warps)
                row = {
                    "name": name,
                    "precision": precision,
                    "shape": [bm, bn, bk, warps],
                    "ms": milliseconds(run),
                    "error": compare(c, reference),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps({"size": n, "rows": m, "cases": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
