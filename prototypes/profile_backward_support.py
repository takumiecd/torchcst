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
        "group_skips": {},
    }
    tile_shapes = (
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (16, 32),
        (16, 64),
        (8, 64),
        (8, 32),
        (8, 16),
        (4, 16),
    )
    for row_tile, col_tile in tile_shapes:
        summary["tiles"][f"{row_tile}x{col_tile}"] = {
            "total": 0,
            "outside": 0,
            "inside": 0,
            "boundary": 0,
            "supported_pairs_inside": 0,
            "supported_pairs_boundary": 0,
            "box_hit": 0,
        }
    for shape in ("16x32", "8x16"):
        summary["group_skips"][shape] = {
            order: {"groups": 0, "exact_outside": 0, "box_outside": 0}
            for order in ("packed", "angle_sorted")
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
        row_start = station * 64
        col_start = 0
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
        for row_tile, col_tile in tile_shapes:
            t = active.reshape(
                count, 64 // row_tile, row_tile, 64 // col_tile, col_tile
            )
            any_tile = t.any(dim=(2, 4))
            all_tile = t.all(dim=(2, 4))
            total = count * (64 // row_tile) * (64 // col_tile)
            inside = int(all_tile.sum().item())
            outside = total - int(any_tile.sum().item())
            entry = summary["tiles"][f"{row_tile}x{col_tile}"]
            entry["total"] += total
            entry["outside"] += outside
            entry["inside"] += inside
            entry["boundary"] += total - outside - inside
            entry["supported_pairs_inside"] += inside * row_tile * col_tile
            entry["supported_pairs_boundary"] += (
                int(active.sum().item()) - inside * row_tile * col_tile
            )
            if (row_tile, col_tile) in ((16, 32), (8, 16), (16, 64), (8, 64)):
                sites_by_tile = sites.reshape(
                    64 // row_tile, row_tile, 64 // col_tile, col_tile, 4
                ).permute(0, 2, 1, 3, 4)
                lower = sites_by_tile.amin(dim=(2, 3))
                upper = sites_by_tile.amax(dim=(2, 3))
                center = atoms[:, None, None, 2:]
                delta = torch.maximum(lower[None] - center, center - upper[None])
                box_dist = delta.clamp_min(0).square().sum(-1)
                box_hit = box_dist * atoms[:, None, None, 1] <= 1.0
                assert bool((any_tile & ~box_hit).any().item()) is False
                entry["box_hit"] += int(box_hit.sum().item())
            if (row_tile, col_tile) in ((16, 32), (8, 16)):
                begin = 0
                for bucket in buckets:
                    length = offsets_cpu[bucket + 1] - offsets_cpu[bucket]
                    stop = begin + length
                    if length:
                        relative = atoms[begin:stop, 2:4]
                        angle = torch.atan2(
                            directions[0, 0] * relative[:, 1]
                            - directions[0, 1] * relative[:, 0],
                            directions[0, 0] * relative[:, 0]
                            + directions[0, 1] * relative[:, 1],
                        )
                        for order, permutation in (
                            ("packed", torch.arange(length, device="cuda")),
                            ("angle_sorted", torch.argsort(angle)),
                        ):
                            exact = any_tile[begin:stop][permutation]
                            box = box_hit[begin:stop][permutation]
                            padding = (-length) % 8
                            if padding:
                                exact = torch.cat(
                                    (
                                        exact,
                                        exact.new_zeros((padding, *exact.shape[1:])),
                                    )
                                )
                                box = torch.cat(
                                    (box, box.new_zeros((padding, *box.shape[1:])))
                                )
                            exact_groups = exact.reshape(-1, 8, *exact.shape[1:]).any(1)
                            box_groups = box.reshape(-1, 8, *box.shape[1:]).any(1)
                            target = summary["group_skips"][f"{row_tile}x{col_tile}"][
                                order
                            ]
                            target["groups"] += exact_groups.numel()
                            target["exact_outside"] += int((~exact_groups).sum().item())
                            target["box_outside"] += int((~box_groups).sum().item())
                    begin = stop
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
