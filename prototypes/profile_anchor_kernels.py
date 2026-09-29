"""Profile one CUDA Graph replay of trainable anchor-sampled CST."""

import argparse
import json
from pathlib import Path

import torch

from prototypes._profile_trace import kernel_summary
from prototypes.anchor_atom_training import anchor_trainable, make_anchor_layout
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--anchor-rows", type=int, choices=(8, 12, 16, 24), default=16)
    parser.add_argument(
        "--anchor-column-segments", type=int, choices=(4, 8, 12), default=8
    )
    parser.add_argument("--basis-mode", choices=("dense", "block"), default="dense")
    parser.add_argument("--backward-lanes", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    layout = make_anchor_layout(layer, args.anchor_rows, args.anchor_column_segments)
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn(args.rows, n, device="cuda", requires_grad=True)
    target = torch.randn(args.rows, n, device="cuda")
    optimizer = torch.optim.AdamW(
        [layer.strip.atoms.p], lr=1e-3, foreach=True, capturable=True
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        output = anchor_trainable(
            layer,
            x,
            layout,
            boxes=boxes,
            witness_cols=hints,
            basis_mode=args.basis_mode,
            backward_lanes=args.backward_lanes,
        )
        (output - target).square().mean().backward()
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "rows": args.rows,
        "anchor_rows": args.anchor_rows,
        "anchor_cols": args.anchor_column_segments * 4,
        "basis_mode": args.basis_mode,
        "backward_lanes": args.backward_lanes,
        "steps": 1,
        **kernel_summary(trace, steps=1),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "kernels"}))
    for entry in result["kernels"][:12]:
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
