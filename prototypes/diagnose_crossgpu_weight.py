"""Compare bounded CST weight windows with the canonical dense oracle."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan, prepare


def compare(actual, expected):
    error = (actual - expected).abs()
    tolerance = 3e-5 + 3e-5 * expected.abs()
    flat = error.flatten()
    largest = int(flat.argmax().item())
    return {
        "max_abs": float(flat[largest].item()),
        "max_location": [largest // actual.shape[1], largest % actual.shape[1]],
        "max_actual": float(actual.flatten()[largest].item()),
        "max_expected": float(expected.flatten()[largest].item()),
        "relative_l2": float((error.norm() / expected.norm()).item()),
        "violations": int((error > tolerance).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--disable-fusion", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    with torch.no_grad():
        reference, canonical = mapped_control(
            layer,
            prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
            canonical_chunk=4,
        )
        plan = execution_plan(layer.strip)
        packed, circle, section, offsets = trainable_boxed_prepare(
            layer.strip,
            layer.strip.atoms.p,
            boxes=station_site_boxes(plan.circle, plan.section, 64),
            witness_cols=balanced_home_columns(layer.strip),
        )
        lists, counts, max_candidates = build_listed_forward_candidates(
            layer, packed, circle, section, offsets
        )
        listed = torch.empty((1024, n), device="cuda")
        unlisted = torch.empty_like(listed)
        profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
        windows = []
        for start in range(0, n, 1024):
            rows = min(1024, n - start)
            materialize_listed[(rows // 64 * layer.column_groups * 4,)](
                packed,
                circle,
                section,
                lists,
                counts,
                offsets,
                listed[:rows],
                K=n,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=profile,
                MAX_CANDIDATES=max_candidates,
                STATION_START=start // 64 * layer.column_groups,
                ROW_START=start,
                num_warps=1,
                enable_fp_fusion=not args.disable_fusion,
            )
            materialize_logical[(rows // 64, layer.column_groups, 2)](
                packed,
                circle,
                section,
                offsets,
                unlisted[:rows],
                N=n,
                K=n,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                D=4,
                PROFILE=profile,
                BN=64,
                BK=32,
                BA=1,
                FACTORED=True,
                ROW_GROUP_START=start // 64,
                LOCAL_W=True,
                num_warps=4,
                enable_fp_fusion=not args.disable_fusion,
            )
            row = {
                "start": start,
                "listed_vs_dense": compare(
                    listed[:rows], reference[start : start + rows]
                ),
                "unlisted_vs_dense": compare(
                    unlisted[:rows], reference[start : start + rows]
                ),
                "listed_vs_unlisted": compare(listed[:rows], unlisted[:rows]),
            }
            windows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "size": n,
        "enable_fp_fusion": not args.disable_fusion,
        "canonical": canonical,
        "windows": windows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
