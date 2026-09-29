"""Compare bounded half-W caching against the current half-window FP32 cache."""

import argparse
import json
from pathlib import Path

import torch

from experiments.cuda.linear.block_streamed_backward import mapped_streamed_trainable
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan


def metrics(actual, reference):
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "allclose_1e3": bool(torch.allclose(actual, reference, atol=1e-3, rtol=1e-3)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--weight-tile-rows", type=int, choices=(8, 16), default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn(args.rows, n, device="cuda")
    dy = torch.randn(args.rows, n, device="cuda")
    results = {}
    for name, cache_windows, cache_dtype, weight_tile_rows in (
        ("fp32_half", 1, torch.float32, 16),
        ("fp16_full", 2, torch.float16, args.weight_tile_rows),
    ):
        layer.strip.atoms.p.grad = None
        xi = x.clone().requires_grad_()
        y = mapped_streamed_trainable(
            layer,
            xi,
            boxes=boxes,
            witness_cols=hints,
            window_rows=512,
            cache_windows=cache_windows,
            cache_weight_dtype=cache_dtype,
            weight_tile_rows=weight_tile_rows,
            atom_kernel="staged_listed",
            gemm_mode="ieee",
            forward_gemm_mode="ieee",
            materialize_mode="listed_bounded",
            listed_unroll=4,
            listed_builder_ba=32,
            listed_builder_warps=1,
        )
        y.backward(dy)
        torch.cuda.synchronize()
        results[name] = {
            "output": y.detach().clone(),
            "input_gradient": xi.grad.detach().clone(),
            "atom_gradient": layer.strip.atoms.p.grad.detach().clone(),
        }
    reference = results["fp32_half"]
    alternative = results["fp16_full"]
    summary = {
        "device": torch.cuda.get_device_name(),
        "rows": args.rows,
        "candidate_weight_tile_rows": args.weight_tile_rows,
        "output": metrics(alternative["output"], reference["output"]),
        "input_gradient": metrics(
            alternative["input_gradient"], reference["input_gradient"]
        ),
        "atom_gradient": metrics(
            alternative["atom_gradient"], reference["atom_gradient"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
