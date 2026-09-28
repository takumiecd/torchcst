"""Compare local-window FP16x3 with TF32x3 and IEEE GEMM on Ada."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton as tr
from triton.runtime.errors import OutOfResources

from prototypes.bounded_gemm import bounded_gemm_kernel
from prototypes.bounded_gemm_fp16x3 import bounded_gemm_fp16x3_kernel


def event_ms(fn, repeats=9):
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


def check(actual, expected):
    diff = (actual - expected).abs()
    tol = 3e-5 + 3e-5 * expected.abs()
    return {
        "max_abs": diff.max().item(),
        "relative_l2": (diff.norm() / expected.norm()).item(),
        "violations_3e_5": (diff > tol).sum().item(),
        "elements": actual.numel(),
    }


def launch(kernel, a, b, c, bm, bn, bk, warps, mode):
    kernel[(tr.cdiv(a.shape[0], bm), tr.cdiv(b.shape[1], bn))](
        a,
        b,
        c,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        *a.stride(),
        *b.stride(),
        *c.stride(),
        bm,
        bn,
        bk,
        False,
        num_warps=warps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, 1024, device="cuda")
    w = torch.randn(1024, n, device="cuda") * 0.01
    cases = {
        "forward": (x, w.T),
        "input_gradient": (dy, w),
    }
    shapes = ((32, 128, 32, 4), (32, 128, 64, 4), (64, 128, 32, 4), (64, 128, 64, 8))
    rows = []
    for name, (a, b) in cases.items():
        reference = a @ b
        c = torch.empty_like(reference)
        for bm, bn, bk, warps in shapes:
            for mode, kernel in (
                ("tf32x3", bounded_gemm_kernel),
                ("fp16x3", bounded_gemm_fp16x3_kernel),
            ):
                run = partial(launch, kernel, a, b, c, bm, bn, bk, warps, mode)

                try:
                    elapsed = event_ms(run)
                except OutOfResources as exc:
                    row = {
                        "case": name,
                        "mode": mode,
                        "tile": [bm, bn, bk, warps],
                        "error": str(exc),
                    }
                else:
                    row = {
                        "case": name,
                        "mode": mode,
                        "tile": [bm, bn, bk, warps],
                        "ms": elapsed,
                        "check": check(c, reference),
                    }
                print(json.dumps(row), flush=True)
                rows.append(row)
    args.output.write_text(
        json.dumps({"size": n, "rows": m, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
