"""Screen substation candidate-list tile shapes on a 1024² mapped layer."""

# The three closures run and are benchmarked before the loop advances.
# ruff: noqa: B023

import argparse
import json
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from experiments.cuda.linear.block_materialize_listed import materialize_listed
from experiments.cuda.linear.block_streamed_backward import trainable_boxed_prepare
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.block_tile_atom_lists import (
    build_tile_atom_lists,
    mapped_backward_atoms_listed,
)
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst._backends.torch.operators.strip_torus.preparation import (
    PROFILE_KINDS,
    execution_plan,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    p, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
    )
    g = layer.strip.chart.tile_count
    cg = layer.column_groups
    profile = PROFILE_KINDS[layer.strip.kernel.profiles[0].binding.profile.id]
    w = torch.empty((512, n), device="cuda")
    dw = torch.randn_like(w)
    dp = torch.zeros_like(p)
    reference_w = None
    reference_dp = None
    results = []
    for br, bc in ((16, 64), (8, 64), (16, 32), (8, 32), (4, 32), (8, 16)):
        tiles = 4096 // (br * bc)
        capacity = 192
        atom_lists = torch.empty((g, tiles, capacity), device="cuda", dtype=torch.uint8)
        counts = torch.empty((g, tiles), device="cuda", dtype=torch.int32)

        def build():
            build_tile_atom_lists[(g, tiles)](
                p,
                circle,
                section,
                offsets,
                atom_lists,
                counts,
                G=g,
                MAX_CANDIDATES=capacity,
                BA=32,
                COMPACT=True,
                BR=br,
                BC=bc,
                BOUNDED=True,
                RANK_LIMIT=256,
                num_warps=1,
                enable_fp_fusion=False,
            )

        def materialize():
            materialize_listed[(512 // 64 * cg * tiles,)](
                p,
                circle,
                section,
                atom_lists,
                counts,
                offsets,
                w,
                K=n,
                CG=cg,
                G=g,
                PROFILE=profile,
                MAX_CANDIDATES=capacity,
                STATION_START=0,
                ROW_START=0,
                BOUNDED=True,
                BR=br,
                BC=bc,
                LOOP_UNROLL=4,
                num_warps=1,
                enable_fp_fusion=False,
            )

        def backward_atoms():
            mapped_backward_atoms_listed[(512 // 64 * cg * (64 // br), 64 // bc)](
                dw,
                p,
                circle,
                section,
                atom_lists,
                counts,
                offsets,
                dp,
                K=n,
                CG=cg,
                G=g,
                PROFILE=profile,
                MAX_CANDIDATES=capacity,
                BA=1,
                COMPACT=True,
                BR=br,
                BC=bc,
                STATION_START=0,
                ROW_START=0,
                OPT_TRIWEIGHT=True,
                BOUNDED=True,
                num_warps=1,
                enable_fp_fusion=True,
            )

        build()
        counts_cpu = counts.cpu()
        materialize()
        torch.cuda.synchronize()
        if reference_w is None:
            reference_w = w.clone()
        w_max_abs = float((w - reference_w).abs().max())
        dp.zero_()
        backward_atoms()
        torch.cuda.synchronize()
        if reference_dp is None:
            reference_dp = dp.clone()
        dp_max_abs = float((dp - reference_dp).abs().max())
        dp_relative_l2 = float(
            (dp - reference_dp).norm() / reference_dp.norm().clamp_min(1e-30)
        )
        result = {
            "tile": [br, bc],
            "tiles_per_station": tiles,
            "candidate_count_min": int(counts_cpu.min()),
            "candidate_count_max": int(counts_cpu.max()),
            "candidate_count_mean": float(counts_cpu[counts_cpu >= 0].float().mean()),
            "fallback_tiles": int((counts_cpu < 0).sum()),
            "w_max_abs": w_max_abs,
            "dp_max_abs": dp_max_abs,
            "dp_relative_l2": dp_relative_l2,
            "build_us": do_bench_cudagraph(build, rep=100) * 1000,
            "materialize_us": do_bench_cudagraph(materialize, rep=100) * 1000,
            "atom_backward_us": do_bench_cudagraph(backward_atoms, rep=100) * 1000,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "results": results}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
