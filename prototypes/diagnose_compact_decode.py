"""Compare Torch and fused intrinsic Torus center decoding at 5% density."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    boxed_prepare,
    decode_intrinsic_torus_compact,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    center = site.atoms.p[:, 2:]
    reference = site.chart.geometry.decode_centers(center)
    compact = decode_intrinsic_torus_compact(site.chart.geometry, center)
    error = (compact - reference).abs()
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    ordinary = boxed_prepare(
        site, site.atoms.p, boxes=boxes, witness_cols=hints, fast_witness=True
    )
    fast = boxed_prepare(
        site,
        site.atoms.p,
        boxes=boxes,
        witness_cols=hints,
        fast_witness=True,
        fast_decode=True,
    )
    packed_error = (ordinary[0] - fast[0]).abs()
    result = {
        "source_commit": args.source_commit,
        "size": n,
        "atoms": center.shape[0],
        "major_radius": site.chart.geometry.major_radius.item(),
        "center_max_abs_by_dim": error.amax(0).tolist(),
        "center_mean_abs_by_dim": error.mean(0).tolist(),
        "center_any_nonfinite": bool(~torch.isfinite(compact).all()),
        "offset_equal": bool(torch.equal(ordinary[3], fast[3])),
        "offset_differences": int((ordinary[3] != fast[3]).sum()),
        "packed_max_abs": packed_error.max().item(),
        "packed_differing_values": int((packed_error > 1e-5).sum()),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
