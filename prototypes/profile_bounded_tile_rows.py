"""Compare bounded-list row-tile sizes on Ada."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--alternate-rows", type=int, choices=(8, 32), default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    cg = layer.column_groups
    rows = min(1024, n)
    capacity = 192
    dw = torch.randn((rows, n), device="cuda")
    tile_rows = (16, args.alternate_rows)
    configs = {}
    for br in tile_rows:
        row_tiles = 64 // br
        lists = torch.empty((g, row_tiles, capacity), device="cuda", dtype=torch.uint8)
        counts = torch.empty((g, row_tiles), device="cuda", dtype=torch.int32)
        w = torch.empty((rows, n), device="cuda")
        dp = torch.zeros_like(packed)
        grid = (rows // 64 * cg * row_tiles,)

        def build(lists=lists, counts=counts, br=br, row_tiles=row_tiles):
            build_tile_atom_lists[(g, row_tiles)](
                packed,
                circle,
                section,
                offsets,
                lists,
                counts,
                G=g,
                MAX_CANDIDATES=capacity,
                BA=8,
                COMPACT=True,
                BR=br,
                BC=64,
                BOUNDED=True,
                RANK_LIMIT=256,
                num_warps=4,
                enable_fp_fusion=False,
            )

        def weight(br=br, grid=grid, lists=lists, counts=counts, w=w):
            materialize_listed[grid](
                packed,
                circle,
                section,
                lists,
                counts,
                offsets,
                w,
                K=n,
                CG=cg,
                G=g,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=capacity,
                STATION_START=0,
                ROW_START=0,
                BOUNDED=True,
                BR=br,
                num_warps=1,
                enable_fp_fusion=False,
            )

        def gradient(br=br, grid=grid, lists=lists, counts=counts, dp=dp):
            dp.zero_()
            mapped_backward_atoms_listed[(grid[0], 1)](
                dw,
                packed,
                circle,
                section,
                lists,
                counts,
                offsets,
                dp,
                K=n,
                CG=cg,
                G=g,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=capacity,
                BA=1,
                COMPACT=True,
                BR=br,
                BC=64,
                STATION_START=0,
                ROW_START=0,
                OPT_TRIWEIGHT=True,
                BOUNDED=True,
                num_warps=1,
                enable_fp_fusion=True,
            )

        build()
        weight()
        gradient()
        torch.cuda.synchronize()
        configs[br] = {
            "build": build,
            "weight": weight,
            "gradient": gradient,
            "lists": lists,
            "counts": counts,
            "w": w,
            "dp": dp,
        }

    w_error = (configs[args.alternate_rows]["w"] - configs[16]["w"]).abs()
    dp_error = (configs[args.alternate_rows]["dp"] - configs[16]["dp"]).abs()
    dp_tolerance = 3e-4 + 3e-4 * configs[16]["dp"].abs()
    checks = {
        "weight_max_abs": w_error.max().item(),
        "weight_violations_3e_5": (w_error > 3e-5 + 3e-5 * configs[16]["w"].abs())
        .sum()
        .item(),
        "gradient_max_abs": dp_error.max().item(),
        "gradient_relative_l2": (
            dp_error.norm() / configs[16]["dp"].norm().clamp_min(1e-30)
        ).item(),
        "gradient_violations_3e_4": (dp_error > dp_tolerance).sum().item(),
    }
    samples = {
        f"{br}_{part}": []
        for br in tile_rows
        for part in ("build", "weight", "gradient")
    }
    for repeat in range(args.rounds):
        for part in ("build", "weight", "gradient"):
            for br in tile_rows if repeat % 2 == 0 else tuple(reversed(tile_rows)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                configs[br][part]()
                end.record()
                end.synchronize()
                samples[f"{br}_{part}"].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": rows,
        "checks": checks,
        "cases": {
            str(br): {
                "list_bytes": configs[br]["lists"].numel(),
                "overflows": (configs[br]["counts"] < 0).sum().item(),
                "median_ms": {
                    part: statistics.median(samples[f"{br}_{part}"])
                    for part in ("build", "weight", "gradient")
                },
            }
            for br in tile_rows
        },
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
