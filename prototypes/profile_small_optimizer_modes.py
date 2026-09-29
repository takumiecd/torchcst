"""Alternate complete bounded CST Graph steps using foreach and fused AdamW."""

import argparse
import copy
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
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(21)
    n = 1024
    first = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    second = copy.deepcopy(first)
    layers = {"foreach": first, "fused": second}
    plan = execution_plan(first.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(first.strip)
    x = {name: torch.randn(args.rows, n, device="cuda", requires_grad=True) for name in layers}
    with torch.no_grad():
        x["fused"].copy_(x["foreach"])
    dy = torch.randn(args.rows, n, device="cuda")
    optimizers = {
        name: torch.optim.AdamW(
            [layer.strip.atoms.p], lr=1e-3, capturable=True, **{name: True}
        )
        for name, layer in layers.items()
    }
    gemm_mode = "ieee" if args.rows == 128 else "fp16x3_dx"
    forward_mode = "ieee" if args.rows == 128 else "fp16x3"

    def forward(name):
        return mapped_streamed_trainable(
            layers[name],
            x[name],
            boxes=boxes,
            witness_cols=hints,
            window_rows=512,
            cache_windows=1,
            atom_kernel="staged_listed",
            gemm_mode=gemm_mode,
            forward_gemm_mode=forward_mode,
            materialize_mode="listed_bounded",
            listed_unroll=4,
            listed_builder_ba=32,
            listed_builder_warps=1,
        )

    with torch.no_grad():
        initial_difference = (forward("foreach") - forward("fused")).abs().max().item()
    assert initial_difference <= 3e-5, initial_difference

    def step(name):
        optimizers[name].zero_grad(set_to_none=True)
        x[name].grad = None
        forward(name).backward(dy)
        optimizers[name].step()

    for _ in range(3):
        step("foreach")
        step("fused")
    torch.cuda.synchronize()
    graphs = {}
    for name in layers:
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            step(name)
    torch.cuda.synchronize()

    samples = {name: [] for name in layers}
    for index in range(args.rounds):
        order = ("foreach", "fused") if index % 2 == 0 else ("fused", "foreach")
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[name].replay()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": args.rows,
        "size": n,
        "rounds": args.rounds,
        "initial_output_max_abs": initial_difference,
        "foreach_median_ms": statistics.median(samples["foreach"]),
        "fused_median_ms": statistics.median(samples["fused"]),
        "paired_fused_foreach_ratio": statistics.median(
            fused / foreach for fused, foreach in zip(samples["fused"], samples["foreach"])
        ),
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples_ms"}))


if __name__ == "__main__":
    main()
