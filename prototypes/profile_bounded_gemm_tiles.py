"""Sweep TF32x3 tile shapes for bounded forward and input-gradient GEMMs."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton as tr

from prototypes.bounded_gemm import bounded_gemm_kernel


def milliseconds(fn):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


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
    w = torch.randn(1024, n, device="cuda")
    forward_output = torch.empty((m, n), device="cuda")
    cases = {
        "forward": (x, w.T, forward_output[:, :1024]),
        "input_gradient": (dy, w, torch.empty_like(x)),
    }
    shapes = (
        (32, 128, 32, 4),
        (32, 128, 64, 4),
        (32, 256, 32, 4),
        (32, 256, 64, 8),
        (64, 64, 32, 4),
        (64, 128, 32, 4),
        (64, 128, 64, 4),
        (64, 128, 64, 8),
        (64, 256, 32, 8),
        (64, 256, 64, 8),
        (128, 64, 32, 8),
        (128, 128, 32, 8),
    )
    results = []
    for name, (a, b, c) in cases.items():
        reference = a @ b
        for bm, bn, bk, warps in shapes:

            def run(a=a, b=b, c=c, bm=bm, bn=bn, bk=bk, warps=warps):
                bounded_gemm_kernel[(tr.cdiv(a.shape[0], bm), tr.cdiv(b.shape[1], bn))](
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

            ms = milliseconds(run)
            difference = (c - reference).abs()
            tolerance = 3e-4 + 3e-4 * reference.abs()
            row = {
                "name": name,
                "shape": [bm, bn, bk, warps],
                "ms": ms,
                "max_abs": difference.max().item(),
                "violations": (difference > tolerance).sum().item(),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps({"size": n, "rows": m, "cases": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
