"""Probe in-place W output from the listed atom-gradient kernel."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import weight_fp_fusion_enabled
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import mapped_backward_atoms_listed
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
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
    lists, counts, max_candidates = build_listed_forward_candidates(
        layer, packed, circle, section, offsets
    )
    chunk = 1024
    grid = (chunk // 64 * layer.column_groups * 4,)
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    dw_initial = torch.randn((chunk, n), device="cuda")
    dw = torch.empty_like(dw_initial)
    reference_w = torch.empty_like(dw_initial)
    dp = torch.zeros_like(packed)

    def materialize():
        return materialize_listed[grid](
            packed, circle, section, lists, counts, offsets, reference_w,
            K=n, CG=layer.column_groups, G=layer.strip.chart.tile_count,
            PROFILE=profile, MAX_CANDIDATES=max_candidates,
            STATION_START=0, ROW_START=0, num_warps=1,
            enable_fp_fusion=weight_fp_fusion_enabled(dw.device),
        )

    def atoms(write_weight, fp_fusion):
        return mapped_backward_atoms_listed[(grid[0], 1)](
            dw, packed, circle, section, lists, counts, offsets, dp,
            K=n, CG=layer.column_groups, G=layer.strip.chart.tile_count,
            PROFILE=profile, MAX_CANDIDATES=max_candidates,
            BA=1, COMPACT=True, BR=16, BC=64,
            STATION_START=0, ROW_START=0, OPT_TRIWEIGHT=True,
            WRITE_WEIGHT=write_weight, num_warps=1,
            enable_fp_fusion=fp_fusion,
        )

    def measure(fn, *, reset_dw=False):
        samples = []
        compiled = None
        for _ in range(6):
            if reset_dw:
                dw.copy_(dw_initial)
                dp.zero_()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            compiled = fn()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        return statistics.median(samples[1:]), compiled

    materialize_ms, _ = measure(materialize)
    baseline_ms, baseline_kernel = measure(
        lambda: atoms(False, True), reset_dw=True
    )
    baseline_dp = dp.clone()
    fused_ms, fused_kernel = measure(
        lambda: atoms(True, False), reset_dw=True
    )
    weight_error = dw - reference_w
    dp_error = dp - baseline_dp
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "window_rows": chunk,
        "max_candidates": max_candidates,
        "materialize_ms": materialize_ms,
        "baseline_atoms_ms": baseline_ms,
        "fused_atoms_and_weight_ms": fused_ms,
        "sum_separate_ms": baseline_ms + materialize_ms,
        "weight_max_abs": weight_error.abs().max().item(),
        "weight_relative_l2": (
            weight_error.norm() / reference_w.norm().clamp_min(1e-30)
        ).item(),
        "dp_max_abs": dp_error.abs().max().item(),
        "dp_relative_l2": (
            dp_error.norm() / baseline_dp.norm().clamp_min(1e-30)
        ).item(),
        "baseline_registers": baseline_kernel.n_regs,
        "baseline_spills": baseline_kernel.n_spills,
        "fused_registers": fused_kernel.n_regs,
        "fused_spills": fused_kernel.n_spills,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
