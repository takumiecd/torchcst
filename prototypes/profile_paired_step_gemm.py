"""Alternate complete mapped CST AdamW steps between bounded GEMM modes."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--control-mode", choices=("ieee", "tf32x3", "tf32x3_dx"), default="tf32x3_dx"
    )
    parser.add_argument(
        "--candidate-mode", choices=("fp16x3", "fp16x3_dx"), default="fp16x3_dx"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn(m, n, device="cuda", requires_grad=True)
    gradient = torch.randn(m, n, device="cuda")
    optimizer = torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, foreach=True)

    def step(mode):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            cache_windows=2,
            atom_kernel="staged_listed",
            gemm_mode=mode,
            forward_gemm_mode=mode.removesuffix("_dx"),
            materialize_mode="listed",
        )
        y.backward(gradient)
        optimizer.step()
        return y

    control, candidate = args.control_mode, args.candidate_mode
    for mode in (control, candidate, candidate, control):
        step(mode)
    torch.cuda.synchronize()
    samples = {control: [], candidate: []}
    for round_index in range(args.rounds):
        order = (control, candidate) if round_index % 2 == 0 else (candidate, control)
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            step(mode)
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "scope": "complete AdamW step, fresh preparation, alternating modes on one updating model",
        "cases": {
            mode: {"median_ms": statistics.median(values), "samples_ms": values}
            for mode, values in samples.items()
        },
        "paired_ratio_median": statistics.median(
            fast / slow for fast, slow in zip(samples[candidate], samples[control])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
