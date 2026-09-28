"""Measure tile-list capacities without changing the training path."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_csr,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
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
    lists, counts, _, cursor = build_listed_forward_candidates_csr(
        layer, packed, circle, section, offsets
    )
    tile_counts = counts.flatten().float()
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "tile_count": counts.numel(),
        "candidate_quantiles": {
            str(q): torch.quantile(tile_counts, q).item()
            for q in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "overflow_tiles_by_capacity": {
            str(cap): (counts > cap).sum().item()
            for cap in (64, 96, 128, 160, 192, 224, 256, 320, 384)
        },
        "reserved_entries": cursor.item(),
        "allocated_entries": lists.numel(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
