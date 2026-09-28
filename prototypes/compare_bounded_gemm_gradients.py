"""Compare large mapped gradients from IEEE and bounded TF32x3 GEMMs."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def compare(actual, expected):
    diff = (actual - expected).abs()
    tolerance = 3e-4 + 3e-4 * expected.abs()
    strict_tolerance = 3e-5 + 3e-5 * expected.abs()
    return {
        "max_abs": diff.max().item(),
        "relative_l1": (diff.sum() / expected.abs().sum()).item(),
        "relative_l2": (diff.norm() / expected.norm()).item(),
        "violations_3e_5": (diff > strict_tolerance).sum().item(),
        "violations_3e_4": (diff > tolerance).sum().item(),
        "elements": actual.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--compare-mode",
        choices=("ieee", "tf32x3", "tf32x3_dx", "fp16x3", "fp16x3_dx", "fp16x4_dw"),
        default="tf32x3",
    )
    parser.add_argument(
        "--forward-gemm-mode", choices=("ieee", "tf32x3", "fp16x3"), default="ieee"
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("default", "listed", "listed_bounded"),
        default="default",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--listed-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn(m, n, device="cuda", requires_grad=True)
    gradient = torch.randn(m, n, device="cuda")
    gradients = []
    outputs = []
    for gemm_mode in ("ieee", args.compare_mode):
        experimental = len(gradients) == 1
        y = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            atom_kernel="staged_listed",
            cache_windows=2,
            gemm_mode=gemm_mode,
            forward_gemm_mode=args.forward_gemm_mode if experimental else "ieee",
            materialize_mode=args.materialize_mode if experimental else "default",
            listed_unroll=args.listed_unroll if experimental else 1,
        )
        dx, dp = torch.autograd.grad(y, (x, layer.strip.atoms.p), gradient)
        torch.cuda.synchronize()
        gradients.append((dx, dp))
        outputs.append(y.detach())
        print(json.dumps({"stage": gemm_mode}), flush=True)
    ieee, x3 = gradients
    result = {
        "size": n,
        "rows": m,
        "seed": args.seed,
        "listed_unroll": args.listed_unroll,
        "output": compare(outputs[1], outputs[0]),
        "dx": compare(x3[0], ieee[0]),
        "dp": compare(x3[1], ieee[1]),
    }
    print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
