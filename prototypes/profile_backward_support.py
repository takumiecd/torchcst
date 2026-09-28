"""Measure exact support sparsity inside routed CST backward stations."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sample-stations", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
    )
    g = layer.strip.chart.tile_count
    cg = layer.column_groups
    stations = torch.linspace(0, g - 1, min(args.sample_stations, g)).round().int()
    offsets_cpu = offsets.cpu().tolist()
    summary = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "atoms": packed.shape[0],
        "sample_stations": stations.tolist(),
        "routed_atom_station_pairs": 0,
        "atom_site_pairs": 0,
        "supported_pairs": 0,
        "column_interval_visited_pairs": 0,
        "column_active_pairs": 0,
        "tiles": {},
    }
    for tile in (4, 8, 16, 32):
        summary["tiles"][str(tile)] = {
            "total": 0,
            "outside": 0,
            "inside": 0,
            "boundary": 0,
            "supported_pairs_inside": 0,
            "supported_pairs_boundary": 0,
        }
    for station in stations.tolist():
        buckets = (
            [0]
            if g == 1
            else [2 * ((station - 1) % g) + 1, 2 * station, 2 * station + 1]
        )
        ids = torch.cat(
            [
                torch.arange(offsets_cpu[b], offsets_cpu[b + 1], device="cuda")
                for b in buckets
            ]
        )
        if ids.numel() == 0:
            continue
        atoms = packed[ids]
        row_start = station // cg * 64
        col_start = station % cg * 64
        directions = circle[row_start : row_start + 64]
        cross = section[col_start : col_start + 64]
        sites = torch.stack(
            (
                directions[:, None, 0] * cross[None, :, 0],
                directions[:, None, 1] * cross[None, :, 0],
                cross[None, :, 1].expand(64, -1),
                cross[None, :, 2].expand(64, -1),
            ),
            dim=-1,
        )
        squared = ((sites[None] - atoms[:, None, None, 2:]) ** 2).sum(-1)
        active = squared * atoms[:, None, None, 1] < 1.0
        count = len(atoms)
        summary["routed_atom_station_pairs"] += count
        summary["atom_site_pairs"] += count * 4096
        summary["supported_pairs"] += int(active.sum().item())
        any_rows = active.any(dim=1)
        summary["column_active_pairs"] += int(any_rows.sum().item()) * 64
        row_indices = torch.arange(64, device="cuda")[None, :, None]
        first = torch.where(active, row_indices, 64).amin(dim=1)
        last = torch.where(active, row_indices, -1).amax(dim=1)
        span = (last - first + 1).clamp_min(0)
        summary["column_interval_visited_pairs"] += int(span.sum().item())
        for tile in (4, 8, 16, 32):
            t = active.reshape(count, 64 // tile, tile, 64 // tile, tile)
            any_tile = t.any(dim=(2, 4))
            all_tile = t.all(dim=(2, 4))
            total = count * (64 // tile) ** 2
            inside = int(all_tile.sum().item())
            outside = total - int(any_tile.sum().item())
            entry = summary["tiles"][str(tile)]
            entry["total"] += total
            entry["outside"] += outside
            entry["inside"] += inside
            entry["boundary"] += total - outside - inside
            entry["supported_pairs_inside"] += inside * tile * tile
            entry["supported_pairs_boundary"] += (
                int(active.sum().item()) - inside * tile * tile
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
