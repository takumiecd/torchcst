"""Measure one CST Graph step's PyTorch peak allocation in a fresh process."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.anchor_atom_training import anchor_trainable, make_anchor_layout
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "anchor16", "anchor24"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    layout = (
        make_anchor_layout(
            layer,
            16 if args.mode == "anchor16" else 24,
            8 if args.mode == "anchor16" else 12,
        )
        if args.mode != "baseline"
        else None
    )
    x = torch.randn(args.rows, n, device="cuda", requires_grad=True)
    target = torch.randn(args.rows, n, device="cuda")
    optimizer = torch.optim.AdamW(
        [layer.strip.atoms.p], lr=1e-3, foreach=True, capturable=True
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        if args.mode == "baseline":
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=512,
                cache_windows=2,
                cache_weight_dtype=torch.float16,
                weight_tile_rows=8,
                atom_kernel="staged_listed",
                gemm_mode="ieee",
                forward_gemm_mode="ieee",
                materialize_mode="listed_bounded",
                listed_unroll=4,
                listed_builder_ba=32,
                listed_builder_warps=1,
            )
        else:
            output = anchor_trainable(layer, x, layout, boxes=boxes, witness_cols=hints)
        (output - target).square().mean().backward()
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "rows": args.rows,
        "mode": args.mode,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "current_allocated_mib": torch.cuda.memory_allocated() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
