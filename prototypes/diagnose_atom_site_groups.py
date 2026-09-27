"""Oracle and conservative-bound capacity of grouped atom/site culling at 5%."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare


def candidate_atoms(packed, offsets, station, stations):
    buckets = (
        2 * ((station + stations - 1) % stations) + 1,
        2 * station,
        2 * station + 1,
    )
    slices = [packed[offsets[b] : offsets[b + 1]] for b in buckets]
    return torch.cat(slices).cpu().numpy().astype(np.float64)


def nearby_groups(centers, radii, width):
    """Deterministic nearest-neighbor grouping for a diagnostic, not a sorter."""
    unused = np.ones(len(centers), dtype=bool)
    groups = []
    while unused.any():
        seed = int(np.flatnonzero(unused)[0])
        available = np.flatnonzero(unused)
        distance = np.sum((centers[available] - centers[seed]) ** 2, axis=1)
        distance += (radii[available] - radii[seed]) ** 2
        selected = available[np.argsort(distance, kind="stable")[:width]]
        unused[selected] = False
        groups.append(selected)
    return groups


def site_groups(rows, columns):
    return np.array(
        [
            [r * 64 + c for r in range(r0, r0 + rows) for c in range(c0, c0 + columns)]
            for r0 in range(0, 64, rows)
            for c0 in range(0, 64, columns)
        ],
        dtype=np.int64,
    )


def measure(centers, radii, active, sites, site_index, atom_width):
    site_points = sites[site_index]
    site_center = site_points.mean(axis=1)
    site_radius = np.linalg.norm(site_points - site_center[:, None, :], axis=2).max(1)
    site_count = site_index.shape[1]
    exact_out = ~active[:, site_index].any(axis=2)
    center_distance = np.linalg.norm(
        centers[:, None, :] - site_center[None, :, :], axis=2
    )
    individual_out = center_distance > radii[:, None] + site_radius[None, :] + 1e-6
    assert not np.any(individual_out & ~exact_out)

    atom_groups = nearby_groups(centers, radii, atom_width)
    group_centers = np.array([centers[group].mean(axis=0) for group in atom_groups])
    outer_radius = np.array(
        [
            np.max(np.linalg.norm(centers[group] - center, axis=1) + radii[group])
            for group, center in zip(atom_groups, group_centers)
        ]
    )
    inner_radius = np.array(
        [
            np.min(radii[group] - np.linalg.norm(centers[group] - center, axis=1))
            for group, center in zip(atom_groups, group_centers)
        ]
    )
    group_distance = np.linalg.norm(
        group_centers[:, None, :] - site_center[None, :, :], axis=2
    )
    bound_out = group_distance > outer_radius[:, None] + site_radius[None, :] + 1e-6
    bound_inside = group_distance + site_radius[None, :] + 1e-6 < inner_radius[:, None]
    oracle_out = np.array(
        [~active[group][:, site_index].any(axis=(0, 2)) for group in atom_groups]
    )
    oracle_inside = np.array(
        [active[group][:, site_index].all(axis=(0, 2)) for group in atom_groups]
    )
    assert not np.any(bound_out & ~oracle_out)
    assert not np.any(bound_inside & ~oracle_inside)
    weights = np.array([len(group) * site_count for group in atom_groups])[:, None]
    pairs = len(centers) * sites.shape[0]
    return {
        "pairs": pairs,
        "atoms": len(centers),
        "atom_groups": len(atom_groups),
        "site_groups": len(site_index),
        "exact_zero_pairs": int((~active).sum()),
        "atom_site_exact_out_pairs": int(exact_out.sum() * site_count),
        "atom_site_bound_out_pairs": int(individual_out.sum() * site_count),
        "group_exact_out_pairs": int((oracle_out * weights).sum()),
        "group_bound_out_pairs": int((bound_out * weights).sum()),
        "group_exact_inside_pairs": int((oracle_inside * weights).sum()),
        "group_bound_inside_pairs": int((bound_inside * weights).sum()),
        "site_radius_median": float(np.median(site_radius)),
        "group_outer_radius_median": float(np.median(outer_radius)),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--stations", type=int, default=8)
    args = parser.parse_args()
    assert args.size in (4096, 8192) and args.stations > 0
    torch.manual_seed(21)
    atoms = round(args.size * args.size * 0.05)
    layer = BlockStripLinear((args.size, args.size), (64, 64), atoms, device="cuda")
    packed, circle, section, offsets = prepare(
        layer.strip, layer.strip.atoms.p, support_layout=True
    )
    station_count = layer.strip.chart.tile_count
    selected = torch.randperm(
        station_count, generator=torch.Generator().manual_seed(43)
    )[: args.stations].tolist()
    offsets = offsets.cpu().tolist()
    section = section.cpu().numpy().astype(np.float64)
    site_shapes = ((16, 32), (8, 8), (4, 8), (4, 4))
    indices = {f"{r}x{c}": site_groups(r, c) for r, c in site_shapes}
    result = {
        "source_commit": args.source_commit,
        "device": torch.cuda.get_device_name(),
        "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
        "shape": [args.size, args.size],
        "atoms": atoms,
        "station_count": station_count,
        "sampled_stations": selected,
        "method": "exact FP64 oracle on decoded FP32 positions; conservative sphere bounds with 1e-6 slack",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for station in selected:
        atom_data = candidate_atoms(packed, offsets, station, station_count)
        centers, radii = atom_data[:, 2:6], 1 / np.sqrt(atom_data[:, 1])
        directions = circle[station * 64 : (station + 1) * 64].cpu().numpy()
        xy = directions[:, None, :] * section[None, :, :1]
        tail = np.broadcast_to(section[None, :, 1:], (64, 64, 2))
        sites = np.concatenate((xy, tail), axis=2).reshape(4096, 4).astype(np.float64)
        squared = ((centers[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2)
        active = squared * atom_data[:, 1, None] < 1
        case = {"station": station, "atom_count": len(atom_data), "groups": {}}
        for shape, site_index in indices.items():
            for width in (1, 8, 16):
                case["groups"][f"{width}x{shape}"] = measure(
                    centers, radii, active, sites, site_index, width
                )
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"station": station, "atoms": len(atom_data)}), flush=True)
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
