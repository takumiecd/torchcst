"""Measure candidate-list backward for several site tile shapes."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import build_tile_atom_lists, mapped_backward_atoms_listed
from prototypes.profile_listed_tuning import milliseconds
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    layer = BlockStripLinear((n, n), (64, 64), round(.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    counts = offsets[1:] - offsets[:-1]
    ids = torch.arange(g, device="cuda")
    candidate_counts = (
        counts[2 * ((ids + g - 1) % g) + 1] + counts[2 * ids] + counts[2 * ids + 1]
    )
    max_candidates = int(candidate_counts.max().item())
    dtype = torch.uint8 if max_candidates <= 256 else torch.uint16
    rows = 1024
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, rows, device="cuda")
    dw = torch.mm(dy.T, x)
    dp = torch.zeros_like(packed)
    results = []
    reference = None
    for br, bc in ((16, 32), (8, 32), (16, 16), (8, 16), (32, 32), (16, 64)):
        tiles = 4096 // (br * bc)
        lists = torch.empty((g, tiles, max_candidates), device="cuda", dtype=dtype)
        list_counts = torch.empty((g, tiles), device="cuda", dtype=torch.int32)

        def build():
            build_tile_atom_lists[(g, tiles)](
                packed, circle, section, offsets, lists, list_counts,
                G=g, MAX_CANDIDATES=max_candidates, BA=8, COMPACT=True,
                BR=br, BC=bc, num_warps=4, enable_fp_fusion=False,
            )

        build_ms = milliseconds(build)

        def backward():
            dp.zero_()
            mapped_backward_atoms_listed[
                (rows // 64 * layer.column_groups * (64 // br), 64 // bc)
            ](
                dw, packed, circle, section, lists, list_counts, offsets, dp,
                K=n, CG=layer.column_groups, G=g,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=max_candidates, BA=1, COMPACT=True,
                BR=br, BC=bc, STATION_START=0, ROW_START=0,
                num_warps=1, enable_fp_fusion=True,
            )

        ms = milliseconds(backward)
        if reference is None:
            reference = dp.clone()
        error = float((dp - reference).abs().sum().item() / reference.abs().sum().item())
        row = {
            "br": br, "bc": bc, "build_ms": build_ms, "backward_ms": ms,
            "list_mb": lists.numel() * lists.element_size() / 1e6,
            "retained_fraction": float(
                list_counts.sum().item() / (candidate_counts.sum().item() * tiles)
            ),
            "relative_l1": error,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"size": n, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
