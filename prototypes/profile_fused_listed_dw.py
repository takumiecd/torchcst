"""Compare local dW GEMM + atom pullback against a fused listed kernel."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import mapped_backward_atoms_listed
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    chunk = min(1024, n // 2)
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
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
    x = torch.randn((m, n), device="cuda")
    dy = torch.randn((m, n), device="cuda")
    dw = torch.empty((chunk, n), device="cuda")
    dp = torch.zeros_like(packed)
    grid = (chunk // 64 * layer.column_groups * 4, 1)
    common = {
        "K": n,
        "CG": layer.column_groups,
        "G": layer.strip.chart.tile_count,
        "PROFILE": PROFILE_KINDS[type(layer.strip.kernel.profile)],
        "MAX_CANDIDATES": max_candidates,
        "BA": 1,
        "COMPACT": True,
        "BR": 16,
        "BC": 64,
        "STATION_START": 0,
        "ROW_START": 0,
        "OPT_TRIWEIGHT": True,
        "enable_fp_fusion": True,
    }

    def baseline():
        dp.zero_()
        torch.mm(dy[:, :chunk].T, x, out=dw)
        return mapped_backward_atoms_listed[grid](
            dw,
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            dp,
            num_warps=1,
            **common,
        )

    def fused():
        dp.zero_()
        return mapped_backward_atoms_listed[grid](
            dw,
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            dp,
            X=x,
            DY=dy,
            M=m,
            N=n,
            FUSED_DW=True,
            num_warps=4,
            **common,
        )

    baseline()
    torch.cuda.synchronize()
    reference = dp.clone()
    compiled = fused()
    torch.cuda.synchronize()
    delta = (dp - reference).abs()
    checks = {
        "max_abs": delta.max().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "violations": (delta > 3e-4 + 3e-4 * reference.abs()).sum().item(),
    }
    samples = {"baseline": [], "fused": []}
    for index in range(args.rounds):
        order = ("baseline", "fused") if index % 2 == 0 else ("fused", "baseline")
        for mode in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            (baseline if mode == "baseline" else fused)()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": [m, n, n],
        "chunk": chunk,
        "scope": "one local dW window plus listed atom pullback; preparation excluded",
        "checks": checks,
        "fused_registers": compiled.n_regs,
        "fused_spills": compiled.n_spills,
        "median_ms": {key: statistics.median(value) for key, value in samples.items()},
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
