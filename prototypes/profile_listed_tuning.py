"""Sweep listed atom-gradient lane counts on one staged dW window."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(fn, repeats=5):
    fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
    return statistics.median(values)


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
    counts = offsets[1:] - offsets[:-1]
    ids = torch.arange(g, device="cuda")
    max_candidates = int(
        (counts[2 * ((ids + g - 1) % g) + 1] + counts[2 * ids] + counts[2 * ids + 1])
        .max()
        .item()
    )
    lists = torch.empty((g, 8, max_candidates), device="cuda", dtype=torch.int32)
    list_counts = torch.empty((g, 8), device="cuda", dtype=torch.int32)
    build_tile_atom_lists[(g, 8)](
        packed,
        circle,
        section,
        offsets,
        lists,
        list_counts,
        G=g,
        MAX_CANDIDATES=max_candidates,
        BA=8,
        COMPACT=False,
        BR=16,
        BC=32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    rows = 1024
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, rows, device="cuda")
    dw = torch.mm(dy.T, x)
    dp = torch.zeros_like(packed)
    results = []
    reference = None
    for ba in (1, 2, 4, 8):
        for warps in (1, 2, 4, 8):

            def run(ba=ba, warps=warps):
                dp.zero_()
                mapped_backward_atoms_listed[(rows // 64 * layer.column_groups * 4, 2)](
                    dw,
                    packed,
                    circle,
                    section,
                    lists,
                    list_counts,
                    offsets,
                    dp,
                    K=n,
                    CG=layer.column_groups,
                    G=g,
                    PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                    MAX_CANDIDATES=max_candidates,
                    BA=ba,
                    COMPACT=False,
                    BR=16,
                    BC=32,
                    STATION_START=0,
                    ROW_START=0,
                    num_warps=warps,
                    enable_fp_fusion=True,
                )

            ms = milliseconds(run)
            if reference is None:
                reference = dp.clone()
            error = float(
                (dp - reference).abs().sum().item() / reference.abs().sum().item()
            )
            results.append({"ba": ba, "warps": warps, "ms": ms, "relative_l1": error})
            print(json.dumps(results[-1]), flush=True)
    result = {"size": n, "max_candidates": max_candidates, "cases": results}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
