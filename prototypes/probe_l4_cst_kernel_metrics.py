"""Launch one current L4 CST W or atom-gradient kernel for Nsight Compute."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import mapped_backward_atoms_listed
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("weight", "atoms"), required=True)
    parser.add_argument("--rows", type=int, choices=(16, 128), default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    br = 8 if args.kernel == "weight" else 16
    lists, counts, capacity = build_listed_forward_candidates_bounded(
        layer,
        packed,
        circle,
        section,
        offsets,
        max_candidates=192,
        ba=32,
        warps=1,
        br=br,
    )
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    w = torch.empty((512, n), device="cuda")
    dp = torch.zeros_like(packed)
    x = torch.randn(args.rows, n, device="cuda")
    dy = torch.randn(args.rows, n, device="cuda")
    if args.kernel == "atoms":
        torch.mm(dy[:, :512].T, x, out=w)

    def launch():
        if args.kernel == "weight":
            return materialize_listed[(512 // 64 * layer.column_groups * 8,)](
                packed,
                circle,
                section,
                lists,
                counts,
                offsets,
                w,
                K=n,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=profile,
                MAX_CANDIDATES=capacity,
                BOUNDED=True,
                BR=8,
                BC=64,
                LOOP_UNROLL=4,
                STATION_START=0,
                ROW_START=0,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return mapped_backward_atoms_listed[(512 // 64 * layer.column_groups * 4, 1)](
            w,
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            dp,
            K=n,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            PROFILE=profile,
            MAX_CANDIDATES=capacity,
            BOUNDED=True,
            BA=1,
            COMPACT=True,
            BR=16,
            BC=64,
            STATION_START=0,
            ROW_START=0,
            OPT_TRIWEIGHT=True,
            num_warps=1,
            enable_fp_fusion=True,
        )

    launch()
    torch.cuda.synchronize()
    compiled = launch()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "kernel": args.kernel,
        "rows": args.rows,
        "registers_per_thread": compiled.n_regs,
        "spills": compiled.n_spills,
        "candidate_capacity": capacity,
        "candidate_mean": float(counts.float().mean()),
        "candidate_max": int(counts.max()),
        "candidate_overflows": int((counts < 0).sum()),
        "tiles": counts.numel(),
        "candidate_site_visits": int(counts[: counts.shape[0] // 2].sum()) * br * 64,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
