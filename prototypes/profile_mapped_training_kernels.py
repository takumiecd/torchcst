"""Profile CUDA kernels in one native 5% mapped CST training step."""

import argparse
import json
from pathlib import Path

import torch

from prototypes._profile_trace import kernel_summary
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn((m, n), device="cuda", requires_grad=True)
    dy = torch.randn((m, n), device="cuda")
    optimizer = torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, foreach=True)
    window_rows = min(1024, n // 2)
    cache_windows = 0 if n == 1024 else 2
    bounded_mode = m == 2048 and n > 1024

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            window_rows=window_rows,
            cache_windows=cache_windows,
            atom_kernel="staged_listed",
            gemm_mode="tf32x3_dx" if bounded_mode else "ieee",
            forward_gemm_mode="tf32x3" if bounded_mode else "ieee",
            materialize_mode="listed",
        )
        y.backward(dy)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        step()
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "rows": m,
        "window_rows": window_rows,
        "cache_windows": cache_windows,
        "atoms": round(0.05 * n * n),
        "steps": 1,
        **kernel_summary(trace, steps=1),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "kernels"}))
    for entry in result["kernels"][:12]:
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
