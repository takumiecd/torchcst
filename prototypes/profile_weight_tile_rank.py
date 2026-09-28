"""Probe numerical rank of native CST 16x64 weight tiles without full W."""

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
    rows = 1024
    w = torch.empty((rows, n), device="cuda")
    materialize_listed[(rows // 64 * layer.column_groups * 4,)](
        packed, circle, section, lists, counts, offsets, w,
        K=n, CG=layer.column_groups, G=layer.strip.chart.tile_count,
        PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
        MAX_CANDIDATES=max_candidates,
        STATION_START=0, ROW_START=0, num_warps=1,
        enable_fp_fusion=weight_fp_fusion_enabled(w.device),
    )
    tiles = w.reshape(rows // 16, 16, n // 64, 64)
    tiles = tiles.permute(0, 2, 1, 3).reshape(-1, 16, 64)
    indices = torch.linspace(0, tiles.shape[0] - 1, steps=256, device="cuda").long()
    singular = torch.linalg.svdvals(tiles[indices])
    energy = singular.square()
    total = energy.sum(dim=1).clamp_min(1e-30)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "sampled_tiles": len(indices),
        "relative_frobenius_tail": {},
    }
    for rank in (2, 4, 8, 12):
        error = (energy[:, rank:].sum(dim=1) / total).sqrt()
        result["relative_frobenius_tail"][str(rank)] = {
            "median": error.median().item(),
            "p90": torch.quantile(error, 0.9).item(),
            "max": error.max().item(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
