"""Isolated peak-allocation and speed probe for 5% mapped forward paths.

Each mode runs in a separate process. The correctness oracle is freed before
measuring memory so CST paths do not inherit a resident dense weight.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    boxed_prepare,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--weight-chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--mode", choices=("dense", "fused", "stream", "stream_fast"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    if args.weight_chunk_rows < 64 or args.weight_chunk_rows % 64:
        parser.error("weight-chunk-rows must be a positive multiple of 64")
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    n = args.size
    atom_count = round(n * n * 0.05)
    layer = BlockStripLinear((n, n), (64, 64), atom_count, device="cuda")
    x = torch.randn(args.batch, n, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"], canonical
    expected = F.linear(x, weight)
    del prepared

    static_bytes = 0
    if args.mode == "dense":
        fn = lambda: F.linear(x, weight)
        parameter_bytes = weight.numel() * weight.element_size()
        del layer
    else:
        parameter_bytes = (
            layer.strip.atoms.p.numel() * layer.strip.atoms.p.element_size()
        )
        if args.mode == "fused":
            fn = lambda: layer(x, backend="triton_fused")
        elif args.mode == "stream":
            fn = lambda: layer(
                x,
                backend="triton_streamed",
                weight_chunk_rows=args.weight_chunk_rows,
            )
        else:
            site = layer.strip
            plan = execution_plan(site)
            boxes = station_site_boxes(plan.circle, plan.section, 64)
            hints = balanced_home_columns(site)
            static_bytes = (
                boxes.numel() * boxes.element_size()
                + hints.numel() * hints.element_size()
            )

            def fn():
                packed = boxed_prepare(
                    site,
                    site.atoms.p,
                    boxes=boxes,
                    witness_cols=hints,
                    fast_witness=True,
                )
                return layer(
                    x,
                    backend="triton_streamed",
                    prepared=packed,
                    weight_chunk_rows=args.weight_chunk_rows,
                )

    output_check = check(fn(), expected)
    assert output_check["passed"], output_check
    del expected
    if args.mode != "dense":
        del weight
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    del output
    torch.cuda.synchronize()
    time_ms = do_bench_cudagraph(fn, rep=20, return_mode="median")
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "mode": args.mode,
        "shape": [args.batch, n, n],
        "atoms": atom_count,
        "weight_chunk_rows": args.weight_chunk_rows,
        "canonical": canonical,
        "output_check": output_check,
        "parameter_bytes": parameter_bytes,
        "static_box_and_hint_bytes": static_bytes,
        "resident_allocated_bytes": resident_allocated,
        "forward_peak_allocated_bytes": peak_allocated,
        "forward_peak_increment_bytes": peak_allocated - resident_allocated,
        "forward_graph_ms": time_ms,
        "memory_scope": "One mode per process. CUDA allocated tensor bytes include resident model, input, fixed geometry, preparation temporaries, weight windows and output. Correctness oracle is freed before measurement. CUDA context, allocator reserved bytes, backward and optimizer are excluded.",
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "completed",
                "mode": args.mode,
                "peak": peak_allocated,
                "ms": time_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
