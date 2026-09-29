"""Compare sampled, trainable CST atoms with the current 1024-square path."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.anchor_atom_training import (
    anchor_samples,
    anchor_trainable,
    make_anchor_layout,
    make_calibrated_anchor_layout,
)
from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, prepare


def comparison(actual, reference):
    difference = actual - reference
    return {
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-30)),
        "max_abs": float(difference.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--anchor-rows", type=int, choices=(8, 12, 16, 24), default=16)
    parser.add_argument(
        "--anchor-column-segments", type=int, choices=(4, 6, 8, 12), default=8
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--basis-mode", choices=("dense", "block"), default="dense")
    parser.add_argument("--forward-lanes", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--backward-lanes", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--decode-mode", choices=("torch", "fused"), default="torch")
    parser.add_argument(
        "--list-mode", choices=("full_tile", "anchors"), default="full_tile"
    )
    parser.add_argument(
        "--optimizer-mode", choices=("foreach", "fused"), default="foreach"
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    baseline = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    anchor = copy.deepcopy(baseline)
    layout = (
        make_calibrated_anchor_layout(
            anchor, args.calibration, args.anchor_rows, args.anchor_column_segments
        )
        if args.calibration is not None
        else make_anchor_layout(anchor, args.anchor_rows, args.anchor_column_segments)
    )
    plan = execution_plan(baseline.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(baseline.strip)
    row_anchors = layout.row_indices
    col_anchors = layout.col_indices

    with torch.no_grad():
        weight, canonical = mapped_control(
            baseline,
            prepare(baseline.strip, baseline.strip.atoms.p, support_layout=True),
            canonical_chunk=4,
        )
        if not canonical["passed"]:
            raise RuntimeError(f"CST control failed: {canonical}")
        dense_weight = (
            torch.nn.Parameter(weight.detach().clone()) if args.include_dense else None
        )
        blocks = weight.reshape(16, 64, 16, 64).permute(0, 2, 1, 3)
        expected_samples = (
            blocks.index_select(2, torch.tensor(row_anchors, device="cuda"))
            .index_select(3, torch.tensor(col_anchors, device="cuda"))
            .permute(0, 2, 1, 3)
            .reshape(16 * args.anchor_rows, 16 * len(col_anchors))
        )
        sampled = anchor_samples(
            anchor,
            layout,
            boxes=boxes,
            witness_cols=hints,
            forward_lanes=args.forward_lanes,
        )
        sample_check = comparison(sampled, expected_samples)
        del sampled, expected_samples, blocks, weight

    x_baseline = torch.randn(args.rows, n, device="cuda", requires_grad=True)
    x_anchor = x_baseline.detach().clone().requires_grad_()
    x_dense = (
        x_baseline.detach().clone().requires_grad_() if args.include_dense else None
    )
    dy = torch.randn(args.rows, n, device="cuda")
    target = torch.randn(args.rows, n, device="cuda")

    def baseline_output():
        return mapped_streamed_trainable(
            baseline,
            x_baseline,
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

    def anchor_output():
        return anchor_trainable(
            anchor,
            x_anchor,
            layout,
            boxes=boxes,
            witness_cols=hints,
            basis_mode=args.basis_mode,
            forward_lanes=args.forward_lanes,
            backward_lanes=args.backward_lanes,
            decode_mode=args.decode_mode,
            list_mode=args.list_mode,
        )

    def dense_output():
        return x_dense @ dense_weight.T

    baseline_y = baseline_output()
    baseline_y.backward(dy)
    reference_dx = x_baseline.grad.detach().clone()
    reference_dp = baseline.strip.atoms.p.grad.detach().clone()
    anchor_y = anchor_output()
    anchor_y.backward(dy)
    initial = {
        "samples": sample_check,
        "output": comparison(anchor_y, baseline_y),
        "input_gradient": comparison(x_anchor.grad, reference_dx),
        "atom_gradient": comparison(anchor.strip.atoms.p.grad, reference_dp),
    }
    if args.include_dense:
        initial["dense_output"] = comparison(dense_output(), baseline_y)
    del baseline_y, anchor_y, reference_dx, reference_dp
    torch.cuda.synchronize()

    optimizers = {
        "baseline": torch.optim.AdamW(
            [baseline.strip.atoms.p],
            lr=1e-3,
            foreach=args.optimizer_mode == "foreach",
            fused=args.optimizer_mode == "fused",
            capturable=True,
        ),
        "anchor": torch.optim.AdamW(
            [anchor.strip.atoms.p],
            lr=1e-3,
            foreach=args.optimizer_mode == "foreach",
            fused=args.optimizer_mode == "fused",
            capturable=True,
        ),
    }
    if args.include_dense:
        optimizers["dense"] = torch.optim.AdamW(
            [dense_weight],
            lr=1e-3,
            foreach=args.optimizer_mode == "foreach",
            fused=args.optimizer_mode == "fused",
            capturable=True,
        )

    def step(mode):
        optimizers[mode].zero_grad(set_to_none=True)
        if mode == "baseline":
            x_baseline.grad = None
            output = baseline_output()
        elif mode == "anchor":
            x_anchor.grad = None
            output = anchor_output()
        else:
            x_dense.grad = None
            output = dense_output()
        (output - target).square().mean().backward()
        optimizers[mode].step()

    modes = (
        ("baseline", "anchor", "dense")
        if args.include_dense
        else ("baseline", "anchor")
    )
    for mode in (*modes, *reversed(modes)):
        step(mode)
    torch.cuda.synchronize()
    graphs = {}
    for mode in modes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(mode)
        graphs[mode] = graph
    torch.cuda.synchronize()
    samples = {mode: [] for mode in modes}
    for index in range(args.rounds):
        offset = index % len(modes)
        order = modes[offset:] + modes[:offset]
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[mode].replay()
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)

    with torch.no_grad():
        after_baseline = baseline_output()
        after_anchor = anchor_output()
        final = {
            "output": comparison(after_anchor, after_baseline),
            "baseline_loss": float((after_baseline - target).square().mean()),
            "anchor_loss": float((after_anchor - target).square().mean()),
            "atom_parameters": comparison(anchor.strip.atoms.p, baseline.strip.atoms.p),
        }
        if args.include_dense:
            after_dense = dense_output()
            final["dense_loss"] = float((after_dense - target).square().mean())
            final["anchor_vs_dense_output"] = comparison(after_anchor, after_dense)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": args.rows,
        "weight_size": [n, n],
        "atoms": round(0.05 * n * n),
        "anchor_shape": [16 * args.anchor_rows, 16 * len(col_anchors)],
        "anchor_sites_fraction": args.anchor_rows * len(col_anchors) / 4096,
        "basis_mode": args.basis_mode,
        "optimizer_mode": args.optimizer_mode,
        "include_dense": args.include_dense,
        "forward_lanes": args.forward_lanes,
        "backward_lanes": args.backward_lanes,
        "decode_mode": args.decode_mode,
        "list_mode": args.list_mode,
        "initial": initial,
        "graph_rounds": args.rounds,
        "updates_per_mode": args.rounds + 3,
        "seed": args.seed,
        "calibration": str(args.calibration) if args.calibration else None,
        "objective": "fixed random MSE target",
        "graph_ms": {
            mode: {
                "median": statistics.median(samples[mode]),
                "samples": samples[mode],
            }
            for mode in modes
        },
        "paired_anchor_vs_baseline": statistics.median(
            a / b for a, b in zip(samples["anchor"], samples["baseline"])
        ),
        "paired_anchor_vs_dense": (
            statistics.median(
                a / b for a, b in zip(samples["anchor"], samples["dense"])
            )
            if args.include_dense
            else None
        ),
        "after_updates": final,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
