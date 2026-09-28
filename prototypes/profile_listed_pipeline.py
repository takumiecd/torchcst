"""Sweep software pipeline and unroll factors in listed atom contraction."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from prototypes.profile_listed_tuning import milliseconds
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triweight-sweep", action="store_true")
    parser.add_argument("--lane-sweep", action="store_true")
    parser.add_argument("--partial-sweep", action="store_true")
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    bucket_counts = offsets[1:] - offsets[:-1]
    ids = torch.arange(g, device="cuda")
    max_candidates = int(
        (
            bucket_counts[2 * ((ids + g - 1) % g) + 1]
            + bucket_counts[2 * ids]
            + bucket_counts[2 * ids + 1]
        )
        .max()
        .item()
    )
    dtype = torch.uint8 if max_candidates <= 256 else torch.uint16
    lists = torch.empty((g, 4, max_candidates), device="cuda", dtype=dtype)
    counts = torch.empty((g, 4), device="cuda", dtype=torch.int32)
    build_tile_atom_lists[(g, 4)](
        packed,
        circle,
        section,
        offsets,
        lists,
        counts,
        G=g,
        MAX_CANDIDATES=max_candidates,
        BA=8,
        COMPACT=True,
        BR=16,
        BC=64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, 1024, device="cuda")
    dw = torch.mm(dy.T, x)
    dp = torch.zeros_like(packed)
    partial = (
        torch.empty(
            (1024 // 64 * layer.column_groups * 4, max_candidates, 5),
            device="cuda",
        )
        if args.partial_sweep
        else None
    )
    results = []
    reference = None
    if args.partial_sweep:
        configs = ((1, 1, True, 1, 1, False), (1, 1, True, 1, 1, True))
    elif args.lane_sweep:
        configs = tuple(
            (1, 1, True, ba, warps, False) for ba in (1, 2, 4, 8) for warps in (1, 2, 4)
        )
    elif args.triweight_sweep:
        configs = ((1, 1, False, 1, 1, False), (1, 1, True, 1, 1, False))
    else:
        configs = tuple(
            (stages, unroll, False, 1, 1, False)
            for stages, unroll in (
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (1, 4),
                (2, 2),
            )
        )
    for stages, unroll, triweight, ba, warps, write_partial in configs:

        def run(
            stages=stages,
            unroll=unroll,
            triweight=triweight,
            ba=ba,
            warps=warps,
            write_partial=write_partial,
        ):
            dp.zero_()
            return mapped_backward_atoms_listed[
                (1024 // 64 * layer.column_groups * 4, 1)
            ](
                dw,
                packed,
                circle,
                section,
                lists,
                counts,
                offsets,
                dp,
                K=n,
                CG=layer.column_groups,
                G=g,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=max_candidates,
                BA=ba,
                COMPACT=True,
                BR=16,
                BC=64,
                STATION_START=0,
                ROW_START=0,
                PIPE_STAGES=stages,
                LOOP_UNROLL=unroll,
                OPT_TRIWEIGHT=triweight,
                WRITE_PARTIAL=write_partial,
                Partial=partial,
                num_warps=warps,
                enable_fp_fusion=True,
            )

        try:
            kernel = run()
            torch.cuda.synchronize()
            if reference is None:
                reference = dp.clone()
            error = (
                float(
                    (dp - reference).abs().sum().item() / reference.abs().sum().item()
                )
                if not write_partial
                else None
            )
            max_abs = (
                float((dp - reference).abs().max().item())
                if not write_partial
                else None
            )
            ms = milliseconds(run)
            row = {
                "stages": stages,
                "unroll": unroll,
                "optimized_triweight": triweight,
                "ba": ba,
                "warps": warps,
                "write_partial": write_partial,
                "ms": ms,
                "relative_l1": error,
                "max_abs_diff": max_abs,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
            }
        except Exception as exc:  # noqa: BLE001 - retain valid variants
            row = {
                "stages": stages,
                "unroll": unroll,
                "optimized_triweight": triweight,
                "ba": ba,
                "warps": warps,
                "write_partial": write_partial,
                "error": str(exc)[:300],
            }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"size": n, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
