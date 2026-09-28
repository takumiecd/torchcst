"""Alternate CUDA Graphs for foreach and fused AdamW on the same CST shape."""

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
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    base = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    layers = {"foreach": base, "fused": copy.deepcopy(base)}
    plan = execution_plan(base.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(base.strip)
    xs = {
        mode: torch.randn((m, n), device="cuda", requires_grad=True) for mode in layers
    }
    xs["fused"].data.copy_(xs["foreach"].data)
    dy = torch.randn((m, n), device="cuda")
    optimizers = {
        mode: torch.optim.AdamW(
            [layer.strip.atoms.p], lr=1e-3, capturable=True, **{mode: True}
        )
        for mode, layer in layers.items()
    }

    def step(mode):
        layer, x = layers[mode], xs[mode]
        optimizers[mode].zero_grad(set_to_none=True)
        x.grad = None
        y = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            atom_kernel="staged_listed",
            window_rows=1024,
            cache_windows=4,
            gemm_mode="fp16x3_dx",
            forward_gemm_mode="fp16x3",
            materialize_mode="listed_bounded",
        )
        y.backward(dy)
        optimizers[mode].step()

    for mode in ("foreach", "fused", "fused", "foreach"):
        step(mode)
    torch.cuda.synchronize()
    graphs = {}
    for mode in layers:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(mode)
        graphs[mode] = graph
    samples = {mode: [] for mode in layers}
    for i in range(args.rounds):
        for mode in ("foreach", "fused") if i % 2 == 0 else ("fused", "foreach"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[mode].replay()
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)
    parameter_difference = (
        layers["fused"].strip.atoms.p - layers["foreach"].strip.atoms.p
    ).abs()
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "rounds": args.rounds,
        "median_ms": {
            mode: statistics.median(values) for mode, values in samples.items()
        },
        "paired_fused_over_foreach_median": statistics.median(
            fused / foreach
            for fused, foreach in zip(samples["fused"], samples["foreach"])
        ),
        "parameters_max_abs_difference": parameter_difference.max().item(),
        "parameters_relative_l2_difference": (
            parameter_difference.norm()
            / layers["foreach"].strip.atoms.p.norm().clamp_min(1e-30)
        ).item(),
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples_ms"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
