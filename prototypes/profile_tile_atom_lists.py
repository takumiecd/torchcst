"""Measure preparation and atom gradients using conservative tile atom lists."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_streamed_backward import (
    mapped_backward_atoms_factored,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(function, repeats=3):
    function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
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
    station_ids = torch.arange(g, device="cuda")
    if g == 1:
        max_candidates = int(bucket_counts[0].item())
    else:
        candidate_counts = (
            bucket_counts[2 * ((station_ids + g - 1) % g) + 1]
            + bucket_counts[2 * station_ids]
            + bucket_counts[2 * station_ids + 1]
        )
        max_candidates = int(candidate_counts.max().item())
    lists = torch.empty((g, 8, max_candidates), dtype=torch.int32, device="cuda")
    counts = torch.empty((g, 8), dtype=torch.int32, device="cuda")
    rows = min(1024, n // 2)
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, n, device="cuda")
    dw = torch.mm(dy[:, :rows].T, x)
    dp = torch.zeros_like(packed)
    stations = rows // 64 * layer.column_groups

    def baseline():
        dp.zero_()
        mapped_backward_atoms_factored[(stations * 4, 2)](
            dw,
            x,
            dy,
            packed,
            circle,
            section,
            offsets,
            dp,
            M=128,
            N=n,
            K=n,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=g,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BM=16,
            BN=16,
            BK=32,
            BA=8,
            STAGED=True,
            STATION_START=0,
            ROW_START=0,
            num_warps=4,
            enable_fp_fusion=True,
        )

    baseline_ms = milliseconds(baseline)
    reference = dp.clone()

    def build():
        build_tile_atom_lists[(g, 8)](
            packed,
            circle,
            section,
            offsets,
            lists,
            counts,
            G=g,
            MAX_CANDIDATES=max_candidates,
            BA=8,
            num_warps=4,
            enable_fp_fusion=False,
        )

    build_ms = milliseconds(build)

    def listed():
        dp.zero_()
        mapped_backward_atoms_listed[(stations * 4, 2)](
            dw,
            packed,
            circle,
            section,
            lists,
            counts,
            dp,
            K=n,
            CG=layer.column_groups,
            G=g,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            MAX_CANDIDATES=max_candidates,
            BA=8,
            STATION_START=0,
            ROW_START=0,
            num_warps=4,
            enable_fp_fusion=True,
        )

    listed_ms = milliseconds(listed)
    difference = (dp - reference).abs()
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "atoms": packed.shape[0],
        "window_rows": rows,
        "max_candidates": max_candidates,
        "list_bytes": lists.numel() * lists.element_size(),
        "count_bytes": counts.numel() * counts.element_size(),
        "retained_fraction": float(
            counts.sum().item() / (candidate_counts.sum().item() * 8)
        )
        if g > 1
        else None,
        "baseline_window_ms": baseline_ms,
        "list_build_ms": build_ms,
        "listed_window_ms": listed_ms,
        "max_abs_diff": float(difference.max().item()),
        "relative_l1": float(difference.sum().item() / reference.abs().sum().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
