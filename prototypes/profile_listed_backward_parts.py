"""Time the GPU phases of the bounded candidate-list backward."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from prototypes.bounded_gemm import bounded_gemm
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cache-windows", type=int, default=0)
    parser.add_argument("--gemm-mode", choices=("ieee", "tf32x3_dx"), default="ieee")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n, m = args.size, args.rows
    if args.cache_windows < 0 or args.cache_windows * 1024 > n // 2:
        parser.error("cached windows must cover no more than half of W")
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    bucket_counts = offsets[1:] - offsets[:-1]
    ids = torch.arange(g, device="cuda")
    max_candidates = int(
        (
            bucket_counts[2 * ((ids + g - 1) % g) + 1]
            + bucket_counts[2 * ids]
            + bucket_counts[2 * ids + 1]
        )
        .max()
        .item()
    )
    dtype = torch.uint8 if max_candidates <= 256 else torch.uint16
    lists = torch.empty((g, 4, max_candidates), device="cuda", dtype=dtype)
    counts = torch.empty((g, 4), device="cuda", dtype=torch.int32)
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, n, device="cuda")
    dx = torch.zeros_like(x)
    dp = torch.zeros_like(packed)
    chunk = 1024
    w = x.new_empty((chunk, n))
    cache = x.new_empty((args.cache_windows * chunk, n))
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]

    def materialize(start, rows, target):
        materialize_logical[(math.ceil(rows / 64), layer.column_groups, 2)](
            packed,
            circle,
            section,
            offsets,
            target,
            N=n,
            K=n,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=g,
            D=4,
            PROFILE=profile,
            BN=64,
            BK=32,
            BA=1,
            FACTORED=True,
            ROW_GROUP_START=start // 64,
            LOCAL_W=True,
            num_warps=4,
            enable_fp_fusion=True,
        )

    for start in range(0, cache.shape[0], chunk):
        materialize(start, chunk, cache[start : start + chunk])

    def run():
        events = {key: [] for key in ("weight", "input_mm", "list", "dw_mm", "atoms")}

        def record(key, fn):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            fn()
            end.record()
            events[key].append((begin, end))

        dx.zero_()
        dp.zero_()
        start_wall = time.perf_counter()
        for start in range(0, n, chunk):
            rows = min(chunk, n - start)
            if start < cache.shape[0]:
                weight = cache[start : start + rows]
            else:
                record(
                    "weight",
                    lambda start=start, rows=rows: materialize(start, rows, w),
                )
                weight = w[:rows]
            if args.gemm_mode == "tf32x3_dx":
                record(
                    "input_mm",
                    lambda start=start, rows=rows, weight=weight: bounded_gemm(
                        dy[:, start : start + rows], weight, dx, add=start > 0
                    ),
                )
            else:
                record(
                    "input_mm",
                    lambda start=start, rows=rows, weight=weight: dx.addmm_(
                        dy[:, start : start + rows], weight
                    ),
                )

        record(
            "list",
            lambda: build_tile_atom_lists[(g, 4)](
                packed,
                circle,
                section,
                offsets,
                lists,
                counts,
                G=g,
                MAX_CANDIDATES=max_candidates,
                BA=8,
                COMPACT=True,
                BR=16,
                BC=64,
                num_warps=4,
                enable_fp_fusion=False,
            ),
        )

        for start in range(0, n, chunk):
            rows = min(chunk, n - start)
            record(
                "dw_mm",
                lambda start=start, rows=rows: torch.mm(
                    dy[:, start : start + rows].T, x, out=w[:rows]
                ),
            )
            record(
                "atoms",
                lambda start=start, rows=rows: mapped_backward_atoms_listed[
                    (rows // 64 * layer.column_groups * 4, 1)
                ](
                    w,
                    packed,
                    circle,
                    section,
                    lists,
                    counts,
                    offsets,
                    dp,
                    K=n,
                    CG=layer.column_groups,
                    G=g,
                    PROFILE=profile,
                    MAX_CANDIDATES=max_candidates,
                    BA=1,
                    COMPACT=True,
                    BR=16,
                    BC=64,
                    STATION_START=start // 64 * layer.column_groups,
                    ROW_START=start,
                    OPT_TRIWEIGHT=True,
                    num_warps=1,
                    enable_fp_fusion=True,
                ),
            )
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start_wall) * 1000
        durations = {
            key: sum(begin.elapsed_time(end) for begin, end in pairs)
            for key, pairs in events.items()
        }
        return wall_ms, durations

    run()
    samples = [run() for _ in range(5)]
    result = {
        "size": n,
        "rows": m,
        "max_candidates": max_candidates,
        "cache_windows": args.cache_windows,
        "gemm_mode": args.gemm_mode,
        "wall_median_ms": statistics.median(wall for wall, _ in samples),
        "gpu_median_ms": {
            key: statistics.median(durations[key] for _, durations in samples)
            for key in samples[0][1]
        },
        "finite": bool(
            torch.isfinite(dp).all().item() and torch.isfinite(dx).all().item()
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
