"""Measure separate-process Graph training peaks at 8192-square."""

import argparse
import gc
import json
from pathlib import Path

import torch

from benchmarks.cuda.linear.benchmark_tile_study import mapped_control
from experiments.cuda.linear.anchor_atom_training import (
    anchor_trainable,
    make_anchor_layout,
)
from experiments.cuda.linear.block_streamed_backward import mapped_streamed_trainable
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "anchor", "dense"), required=True
    )
    parser.add_argument("--anchor-rows", type=int, choices=(16, 24), default=16)
    parser.add_argument(
        "--anchor-column-segments", type=int, choices=(8, 12), default=8
    )
    parser.add_argument("--backward-lanes", type=int, choices=(4, 8), default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 8192
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    if args.mode == "dense":
        with torch.no_grad():
            weight, canonical = mapped_control(
                layer,
                prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
                canonical_chunk=4,
            )
            if not canonical["passed"]:
                raise RuntimeError(f"dense control failed: {canonical}")
        parameter = torch.nn.Parameter(weight.detach())
        del layer, weight
        gc.collect()
        torch.cuda.empty_cache()
    else:
        parameter = layer.strip.atoms.p
        plan = execution_plan(layer.strip)
        boxes = station_site_boxes(plan.circle, plan.section, 64)
        hints = balanced_home_columns(layer.strip)
        layout = (
            make_anchor_layout(
                layer,
                args.anchor_rows,
                args.anchor_column_segments,
                dense_bases=False,
            )
            if args.mode == "anchor"
            else None
        )
    x = torch.randn(args.rows, n, device="cuda", requires_grad=True)
    target = torch.randn(args.rows, n, device="cuda")
    optimizer = torch.optim.AdamW(
        [parameter], lr=1e-3, foreach=False, fused=True, capturable=True
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        if args.mode == "dense":
            output = x @ parameter.T
        elif args.mode == "anchor":
            output = anchor_trainable(
                layer,
                x,
                layout,
                boxes=boxes,
                witness_cols=hints,
                basis_mode="block",
                forward_lanes=1,
                backward_lanes=args.backward_lanes,
                decode_mode="torch",
                list_mode="anchors",
            )
        else:
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=1024,
                cache_windows=4,
                cache_weight_dtype=torch.float16,
                weight_tile_rows=8,
                atom_kernel="staged_listed",
                gemm_mode="tf32x3" if args.rows == 2048 else "ieee",
                forward_gemm_mode="tf32x3" if args.rows == 2048 else "ieee",
                materialize_mode="listed_bounded",
                listed_unroll=4,
                listed_builder_ba=32,
                listed_builder_warps=1,
            )
        (output - target).square().mean().backward()
        optimizer.step()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "rows": args.rows,
        "mode": args.mode,
        "anchor_shape": (
            [args.anchor_rows, args.anchor_column_segments * 4]
            if args.mode == "anchor"
            else None
        ),
        "capture_start_allocated_mib": start / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "current_allocated_mib": torch.cuda.memory_allocated() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
