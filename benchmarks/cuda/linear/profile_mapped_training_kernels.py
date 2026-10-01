"""Profile CUDA kernels in one native 5% mapped CST training step."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.cuda.linear._profile_trace import kernel_summary
from experiments.cuda.linear.block_streamed_backward import mapped_streamed_trainable
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(16, 128, 2048), required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--cache-windows", type=int, default=None)
    parser.add_argument(
        "--cache-weight-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--weight-tile-rows", type=int, choices=(8, 16), default=16)
    parser.add_argument("--listed-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--listed-builder-ba", type=int, default=8)
    parser.add_argument("--listed-builder-warps", type=int, default=4)
    parser.add_argument(
        "--materialize-mode",
        choices=("listed", "listed_csr", "listed_bounded"),
        default="listed",
    )
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
    optimizer = torch.optim.AdamW(
        [layer.strip.atoms.p], lr=1e-3, foreach=True, capturable=args.graph
    )
    window_rows = min(1024, n // 2)
    cache_windows = (
        (0 if n == 1024 else 2) if args.cache_windows is None else args.cache_windows
    )
    bounded_mode = m == 2048

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
            gemm_mode="fp16x3_dx" if bounded_mode else "ieee",
            forward_gemm_mode="fp16x3" if bounded_mode else "ieee",
            materialize_mode=args.materialize_mode,
            listed_unroll=args.listed_unroll,
            listed_builder_ba=args.listed_builder_ba,
            listed_builder_warps=args.listed_builder_warps,
            cache_weight_dtype=getattr(torch, args.cache_weight_dtype),
            weight_tile_rows=args.weight_tile_rows,
        )
        y.backward(dy)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        run_step = graph.replay
    else:
        run_step = step
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        run_step()
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "rows": m,
        "graph": args.graph,
        "materialize_mode": args.materialize_mode,
        "listed_unroll": args.listed_unroll,
        "listed_builder_ba": args.listed_builder_ba,
        "listed_builder_warps": args.listed_builder_warps,
        "window_rows": window_rows,
        "cache_windows": cache_windows,
        "cache_weight_dtype": args.cache_weight_dtype,
        "weight_tile_rows": args.weight_tile_rows,
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
