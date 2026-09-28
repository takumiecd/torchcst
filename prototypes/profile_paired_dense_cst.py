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
    parser.add_argument(
        "--gemm-mode", choices=("auto", "ieee", "fp16x3_dx"), default="auto"
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("listed", "listed_parallel", "listed_csr"),
        default="listed",
    )
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--baseline-same-gemm", action="store_true")
    parser.add_argument("--graph", action="store_true")
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
    window_rows = min(1024, n // 2)
    cache_windows = 0 if n == 1024 else 2
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
        )
        accuracy = check(actual, expected)
        assert accuracy["passed"], accuracy
    del actual, expected
    optimizers = {
        "dense": torch.optim.AdamW(
            [dense_weight], lr=1e-3, foreach=True, capturable=args.graph
        ),
        "cst": torch.optim.AdamW(
            [layer.strip.atoms.p], lr=1e-3, foreach=True, capturable=args.graph
        ),
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
                window_rows=window_rows,
                cache_windows=cache_windows,
                gemm_mode="ieee"
                if baseline and not args.baseline_same_gemm
                else cst_mode,
                forward_gemm_mode=(
                    "ieee" if baseline and not args.baseline_same_gemm else forward_mode
                ),
                materialize_mode="listed" if baseline else args.materialize_mode,
            )
        output.backward(gradient)
        optimizers[key].step()

    modes = (
        ("dense", "cst_baseline", "cst") if args.compare_baseline else ("dense", "cst")
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
        "graph": args.graph,
        "window_rows": window_rows,
        "cache_windows": cache_windows,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
