"""Compare dense and bounded CST complete steps in alternating order on one GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    with torch.no_grad():
        weight, canonical = mapped_control(
            layer,
            prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
            canonical_chunk=4,
        )
    assert canonical["passed"]
    dense_weight = torch.nn.Parameter(weight.detach())
    del weight
    x = torch.randn(m, n, device="cuda", requires_grad=True)
    gradient = torch.randn(m, n, device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    cst_mode = "ieee" if m == 128 else "fp16x3_dx"
    forward_mode = "ieee" if m == 128 else "fp16x3"
    with torch.no_grad():
        expected = F.linear(x, dense_weight)
        actual = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            atom_kernel="staged_listed",
            cache_windows=2,
            gemm_mode=cst_mode,
            forward_gemm_mode=forward_mode,
            materialize_mode="listed",
        )
        accuracy = check(actual, expected)
        assert accuracy["passed"], accuracy
    del actual, expected
    optimizers = {
        "dense": torch.optim.AdamW([dense_weight], lr=1e-3, foreach=True),
        "cst": torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, foreach=True),
    }

    def step(mode):
        optimizers[mode].zero_grad(set_to_none=True)
        x.grad = None
        if mode == "dense":
            output = F.linear(x, dense_weight)
        else:
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                atom_kernel="staged_listed",
                cache_windows=2,
                gemm_mode=cst_mode,
                forward_gemm_mode=forward_mode,
                materialize_mode="listed",
            )
        output.backward(gradient)
        optimizers[mode].step()

    for mode in ("dense", "cst", "cst", "dense"):
        step(mode)
    torch.cuda.synchronize()
    samples = {"dense": [], "cst": []}
    for round_index in range(args.rounds):
        order = ("dense", "cst") if round_index % 2 == 0 else ("cst", "dense")
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
        "scope": "complete AdamW steps; dense and CST weights and optimizer states coexist for timing; separate-process peaks required for memory",
        "accuracy": accuracy,
        "cases": {
            mode: {"median_ms": statistics.median(values), "samples_ms": values}
            for mode, values in samples.items()
        },
        "paired_ratio_median": statistics.median(
            cst / dense for cst, dense in zip(samples["cst"], samples["dense"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
