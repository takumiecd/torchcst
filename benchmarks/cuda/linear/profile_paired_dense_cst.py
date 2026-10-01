"""Compare dense and bounded CST complete steps in alternating order on one GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.cuda.linear.benchmark_large_forward import check
from benchmarks.cuda.linear.benchmark_tile_study import mapped_control
from experiments.cuda.linear.block_streamed_backward import mapped_streamed_trainable
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(16, 128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--window-rows", type=int, default=None)
    parser.add_argument("--cache-windows", type=int, default=None)
    parser.add_argument("--alt-window-rows", type=int, default=None)
    parser.add_argument("--alt-cache-windows", type=int, default=None)
    parser.add_argument(
        "--cache-weight-dtype", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument(
        "--alt-cache-weight-dtype", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument("--weight-tile-rows", type=int, choices=(8, 16), default=16)
    parser.add_argument("--alt-weight-tile-rows", type=int, choices=(8, 16), default=16)
    parser.add_argument(
        "--gemm-mode", choices=("auto", "ieee", "fp16x3_dx"), default="auto"
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("listed", "listed_parallel", "listed_csr", "listed_bounded"),
        default="listed",
    )
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--compare-csr", action="store_true")
    parser.add_argument("--compare-unroll", action="store_true")
    parser.add_argument("--compare-builder", action="store_true")
    parser.add_argument("--listed-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--listed-builder-ba", type=int, default=8)
    parser.add_argument("--listed-builder-warps", type=int, default=4)
    parser.add_argument("--baseline-same-gemm", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--optimizer-mode", choices=("foreach", "fused"), default="foreach"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_alternate = args.alt_window_rows is not None
    if compare_alternate != (args.alt_cache_windows is not None):
        parser.error(
            "alternate window rows and cache windows must be specified together"
        )
    if (
        sum(
            (
                args.compare_baseline,
                args.compare_csr,
                args.compare_unroll,
                args.compare_builder,
                compare_alternate,
            )
        )
        > 1
    ):
        parser.error("choose one comparison mode")
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
    window_rows = min(1024, n // 2) if args.window_rows is None else args.window_rows
    cache_windows = (
        (0 if n == 1024 else 2) if args.cache_windows is None else args.cache_windows
    )
    cst_mode = (
        ("ieee" if m == 128 or n == 1024 else "fp16x3_dx")
        if args.gemm_mode == "auto"
        else args.gemm_mode
    )
    forward_mode = "ieee" if cst_mode == "ieee" else "fp16x3"
    with torch.no_grad():
        expected = F.linear(x, dense_weight)
        actual = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            atom_kernel="staged_listed",
            window_rows=window_rows,
            cache_windows=cache_windows,
            gemm_mode=cst_mode,
            forward_gemm_mode=forward_mode,
            materialize_mode=args.materialize_mode,
            listed_unroll=args.listed_unroll,
            listed_builder_ba=args.listed_builder_ba,
            listed_builder_warps=args.listed_builder_warps,
            cache_weight_dtype=torch.float16
            if args.cache_weight_dtype == "fp16"
            else torch.float32,
            weight_tile_rows=args.weight_tile_rows,
        )
        accuracy = check(actual, expected)
        assert accuracy["passed"], accuracy
    del actual, expected
    optimizer_kwargs = {args.optimizer_mode: True, "capturable": args.graph}
    optimizers = {
        "dense": torch.optim.AdamW([dense_weight], lr=1e-3, **optimizer_kwargs),
        "cst": torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, **optimizer_kwargs),
    }

    def step(mode):
        key = "dense" if mode == "dense" else "cst"
        optimizers[key].zero_grad(set_to_none=True)
        x.grad = None
        if mode == "dense":
            output = F.linear(x, dense_weight)
        else:
            baseline = mode == "cst_baseline"
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                atom_kernel="staged_listed",
                window_rows=(
                    args.alt_window_rows if mode == "cst_alt" else window_rows
                ),
                cache_windows=(
                    args.alt_cache_windows if mode == "cst_alt" else cache_windows
                ),
                cache_weight_dtype=(
                    torch.float16
                    if (
                        args.alt_cache_weight_dtype
                        if mode == "cst_alt"
                        else args.cache_weight_dtype
                    )
                    == "fp16"
                    else torch.float32
                ),
                weight_tile_rows=(
                    args.alt_weight_tile_rows
                    if mode == "cst_alt"
                    else args.weight_tile_rows
                ),
                gemm_mode="ieee"
                if baseline and not args.baseline_same_gemm
                else cst_mode,
                forward_gemm_mode=(
                    "ieee" if baseline and not args.baseline_same_gemm else forward_mode
                ),
                materialize_mode=(
                    "listed"
                    if baseline
                    else "listed_csr"
                    if mode == "cst_csr"
                    else args.materialize_mode
                ),
                listed_unroll=1 if mode == "cst_unroll1" else args.listed_unroll,
                listed_builder_ba=(
                    8 if mode == "cst_builder_baseline" else args.listed_builder_ba
                ),
                listed_builder_warps=(
                    4 if mode == "cst_builder_baseline" else args.listed_builder_warps
                ),
            )
        output.backward(gradient)
        optimizers[key].step()

    modes = (
        ("dense", "cst_baseline", "cst")
        if args.compare_baseline
        else ("dense", "cst_csr", "cst")
        if args.compare_csr
        else ("dense", "cst_unroll1", "cst")
        if args.compare_unroll
        else ("dense", "cst_builder_baseline", "cst")
        if args.compare_builder
        else ("dense", "cst_alt", "cst")
        if compare_alternate
        else ("dense", "cst")
    )
    for mode in (*modes, *reversed(modes)):
        step(mode)
    torch.cuda.synchronize()
    graphs = {}
    if args.graph:
        for mode in modes:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step(mode)
            graphs[mode] = graph
        torch.cuda.synchronize()
    samples = {mode: [] for mode in modes}
    for round_index in range(args.rounds):
        order = modes[round_index % len(modes) :] + modes[: round_index % len(modes)]
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[mode].replay() if args.graph else step(mode)
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "cst_mode": cst_mode,
        "materialize_mode": args.materialize_mode,
        "listed_unroll": args.listed_unroll,
        "listed_builder_ba": args.listed_builder_ba,
        "listed_builder_warps": args.listed_builder_warps,
        "graph": args.graph,
        "optimizer_mode": args.optimizer_mode,
        "window_rows": window_rows,
        "cache_windows": cache_windows,
        "cache_weight_dtype": args.cache_weight_dtype,
        "weight_tile_rows": args.weight_tile_rows,
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
    if args.compare_baseline:
        result["paired_cst_vs_baseline_ratio_median"] = statistics.median(
            cst / baseline
            for cst, baseline in zip(samples["cst"], samples["cst_baseline"])
        )
    if args.compare_csr:
        result["paired_cst_vs_csr_ratio_median"] = statistics.median(
            cst / csr for cst, csr in zip(samples["cst"], samples["cst_csr"])
        )
    if args.compare_unroll:
        result["paired_cst_vs_unroll1_ratio_median"] = statistics.median(
            cst / baseline
            for cst, baseline in zip(samples["cst"], samples["cst_unroll1"])
        )
    if args.compare_builder:
        result["paired_cst_vs_builder_baseline_ratio_median"] = statistics.median(
            cst / baseline
            for cst, baseline in zip(samples["cst"], samples["cst_builder_baseline"])
        )
    if compare_alternate:
        result["alternate"] = {
            "window_rows": args.alt_window_rows,
            "cache_windows": args.alt_cache_windows,
            "cache_weight_dtype": args.alt_cache_weight_dtype,
            "weight_tile_rows": args.alt_weight_tile_rows,
        }
        result["paired_alternate_vs_cst_ratio_median"] = statistics.median(
            alternate / cst
            for alternate, cst in zip(samples["cst_alt"], samples["cst"])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
