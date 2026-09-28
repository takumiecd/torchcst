"""Compare local IEEE dW GEMM with and without a contiguous transposed dY."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--window-rows", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m, chunk = args.size, args.rows, args.window_rows
    x = torch.randn((m, n), device="cuda")
    dy = torch.randn((m, n), device="cuda")
    dy_t = torch.empty((n, m), device="cuda")
    dw = torch.empty((chunk, n), device="cuda")
    reference = torch.empty_like(dw)
    torch.mm(dy[:, :chunk].T, x, out=reference)
    dy_t.copy_(dy.T)
    torch.mm(dy_t[:chunk], x, out=dw)
    difference = (dw - reference).abs()
    checks = {
        "max_abs": difference.max().item(),
        "violations_3e_4": (difference > 3e-4 + 3e-4 * reference.abs()).sum().item(),
    }

    def baseline():
        for start in range(0, n, chunk):
            torch.mm(dy[:, start : start + chunk].T, x, out=dw)

    def transposed():
        dy_t.copy_(dy.T)
        for start in range(0, n, chunk):
            torch.mm(dy_t[start : start + chunk], x, out=dw)

    baseline()
    transposed()
    torch.cuda.synchronize()
    samples = {"baseline": [], "transposed": []}
    for repeat in range(args.rounds):
        order = (
            ("baseline", "transposed")
            if repeat % 2 == 0
            else ("transposed", "baseline")
        )
        for mode in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            (baseline if mode == "baseline" else transposed)()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "window_rows": chunk,
        "transposed_activation_bytes": dy_t.numel() * dy_t.element_size(),
        "checks": checks,
        "median_ms": {
            name: statistics.median(values) for name, values in samples.items()
        },
        "paired_transposed_over_baseline_median": statistics.median(
            alternative / baseline
            for alternative, baseline in zip(samples["transposed"], samples["baseline"])
        ),
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples_ms"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
