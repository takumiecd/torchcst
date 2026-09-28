"""Compare A100 matmul precision modes on bounded backward windows."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def milliseconds(fn, repeats=7):
    fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def compare(actual, expected):
    difference = (actual - expected).abs()
    tolerance = 3e-4 + 3e-4 * expected.abs()
    return {
        "max_abs": float(difference.max().item()),
        "relative_l2": float(difference.norm().item() / expected.norm().item()),
        "violations_3e_4": int((difference > tolerance).sum().item()),
        "elements": actual.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, 1024, device="cuda")
    w = torch.randn(1024, n, device="cuda")
    dw = torch.empty_like(w)
    dx = torch.empty_like(x)
    torch.set_float32_matmul_precision("highest")
    torch.mm(dy.T, x, out=dw)
    torch.mm(dy, w, out=dx)
    reference_dw, reference_dx = dw.clone(), dx.clone()
    results = []
    for mode in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(mode)
        dw_ms = milliseconds(lambda: torch.mm(dy.T, x, out=dw))
        dx_ms = milliseconds(lambda: torch.mm(dy, w, out=dx))
        row = {
            "mode": mode,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "dw_ms": dw_ms,
            "dx_ms": dx_ms,
            "dw_error": compare(dw, reference_dw),
            "dx_error": compare(dx, reference_dx),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    torch.set_float32_matmul_precision("high")
    x_hi = x.to(torch.float16).to(torch.float32)
    dy_hi = dy.to(torch.float16).to(torch.float32)
    w_hi = w.to(torch.float16).to(torch.float32)
    x_lo = x - x_hi
    dy_lo = dy - dy_hi
    w_lo = w - w_hi

    def dw_three():
        torch.mm(dy_hi.T, x_hi, out=dw)
        dw.addmm_(dy_hi.T, x_lo)
        dw.addmm_(dy_lo.T, x_hi)

    def dx_three():
        torch.mm(dy_hi, w_hi, out=dx)
        dx.addmm_(dy_hi, w_lo)
        dx.addmm_(dy_lo, w_hi)

    dw_three()
    dx_three()
    row = {
        "mode": "three_tf32_products",
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "dw_ms": milliseconds(dw_three),
        "dx_ms": milliseconds(dx_three),
        "dw_error": compare(dw, reference_dw),
        "dx_error": compare(dx, reference_dx),
        "split_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in (x_hi, x_lo, dy_hi, dy_lo, w_hi, w_lo)
        ),
    }
    results.append(row)
    print(json.dumps(row), flush=True)

    def dw_four():
        dw_three()
        dw.addmm_(dy_lo.T, x_lo)

    def dx_four():
        dx_three()
        dx.addmm_(dy_lo, w_lo)

    dw_four()
    dx_four()
    row = {
        "mode": "four_tf32_products",
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "dw_ms": milliseconds(dw_four),
        "dx_ms": milliseconds(dx_four),
        "dw_error": compare(dw, reference_dw),
        "dx_error": compare(dx, reference_dx),
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    output = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "torch": torch.__version__,
        "cases": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
