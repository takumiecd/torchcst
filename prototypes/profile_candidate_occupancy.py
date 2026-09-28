"""Measure whether 5% CST leaves enough empty 16x64 tiles to skip GEMM work."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import weight_fp_fusion_enabled
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists, counts, max_candidates = build_listed_forward_candidates(
        layer, packed, circle, section, offsets
    )
    c = counts.float()
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "atoms": round(n * n * 0.05),
        "candidate_tiles": c.numel(),
        "candidate_count": {
            "min": c.min().item(),
            "median": c.median().item(),
            "mean": c.mean().item(),
            "max": c.max().item(),
            "empty_fraction": (c == 0).float().mean().item(),
            "at_most_one_fraction": (c <= 1).float().mean().item(),
        },
        "windows": [],
    }
    # A fully supported atom can potentially enter a tile-level polynomial
    # reduction; a boundary atom still needs the pointwise support test.
    support = {
        shape: {"full": 0, "boundary": 0, "outside": 0}
        for shape in ((16, 64), (8, 32), (8, 16), (4, 16))
    }
    active_site_pairs = 0
    sampled_site_pairs = 0
    samples = 64
    for flat_tile in torch.linspace(
        0, counts.numel() - 1, steps=samples, dtype=torch.long
    ).tolist():
        station, row_tile = divmod(flat_tile, 4)
        count = int(counts[station, row_tile].item())
        ranks = lists[station, row_tile, :count].long()
        g = layer.strip.chart.tile_count
        bucket0 = 2 * ((station + g - 1) % g) + 1
        begin0 = int(offsets[bucket0].item())
        length0 = int((offsets[bucket0 + 1] - offsets[bucket0]).item())
        begin1 = int(offsets[2 * station].item())
        length1 = int((offsets[2 * station + 1] - offsets[2 * station]).item())
        begin2 = int(offsets[2 * station + 1].item())
        atoms = torch.where(
            ranks < length0, begin0 + ranks,
            torch.where(
                ranks < length0 + length1,
                begin1 + ranks - length0,
                begin2 + ranks - length0 - length1,
            ),
        )
        local_rows = station * 64 + row_tile * 16 + torch.arange(16, device="cuda")
        col = torch.arange(64, device="cuda")
        cosine = circle[local_rows, 0]
        sine = circle[local_rows, 1]
        sx = cosine[:, None] * section[col, 0][None, :]
        sy = sine[:, None] * section[col, 0][None, :]
        sites = torch.stack((
            sx, sy,
            section[col, 1].expand_as(sx),
            section[col, 2].expand_as(sx),
        ), dim=-1)
        centers = packed[atoms, 2:6]
        precision = packed[atoms, 1]
        scaled = ((sites[None] - centers[:, None, None, :]) ** 2).sum(-1)
        scaled *= precision[:, None, None]
        active_site_pairs += int((scaled < 1).sum().item())
        sampled_site_pairs += scaled.numel()
        for (tile_rows, tile_cols), record in support.items():
            shaped = scaled.reshape(
                count, 16 // tile_rows, tile_rows, 64 // tile_cols, tile_cols
            )
            hi = shaped.amax(dim=(2, 4))
            lo = shaped.amin(dim=(2, 4))
            record["full"] += int((hi < 1).sum().item())
            record["outside"] += int((lo >= 1).sum().item())
            record["boundary"] += int(((lo < 1) & (hi >= 1)).sum().item())
    result["sampled_candidate_support"] = {}
    result["sampled_atom_site_pairs"] = {
        "pairs": sampled_site_pairs,
        "inside_support_fraction": active_site_pairs / sampled_site_pairs,
        "outside_support_fraction": 1 - active_site_pairs / sampled_site_pairs,
    }
    for shape, record in support.items():
        total = sum(record.values())
        result["sampled_candidate_support"][f"{shape[0]}x{shape[1]}"] = {
            "sampled_parent_tiles": samples,
            "atom_subtiles": total,
            "fully_inside_fraction": record["full"] / total,
            "boundary_fraction": record["boundary"] / total,
            "fully_outside_fraction": record["outside"] / total,
        }
    rows = 1024
    w = torch.empty((rows, n), device="cuda")
    for start in (0, n // 2):
        materialize_listed[(rows // 64 * layer.column_groups * 4,)](
            packed, circle, section, lists, counts, offsets, w,
            K=n, CG=layer.column_groups, G=layer.strip.chart.tile_count,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            MAX_CANDIDATES=max_candidates,
            STATION_START=start // 64 * layer.column_groups,
            ROW_START=start,
            num_warps=1,
            enable_fp_fusion=weight_fp_fusion_enabled(w.device),
        )
        tile_nonzero = w.view(rows // 16, 16, n // 64, 64).ne(0).any(dim=(1, 3))
        result["windows"].append({
            "row_start": start,
            "nonzero_16x64_fraction": tile_nonzero.float().mean().item(),
            "nonzero_element_fraction": w.ne(0).float().mean().item(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
