"""Sweep wide site tiles in the staged atom-gradient kernel."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_streamed_backward import (
    mapped_backward_atoms_factored,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(function, repeats=3):
    function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    rows = min(1024, n // 2)
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, n, device="cuda")
    dw = torch.mm(dy[:, :rows].T, x)
    dp = torch.zeros_like(packed)
    stations = rows // 64 * layer.column_groups
    variants = []
    reference = None
    for bn, bk, ba, warps in (
        (16, 32, 8, 4),
        (32, 32, 4, 4),
        (64, 16, 4, 4),
        (64, 32, 2, 4),
        (64, 64, 1, 4),
        (32, 64, 2, 4),
        (16, 64, 4, 4),
        (64, 16, 4, 8),
    ):

        def run(bn=bn, bk=bk, ba=ba, warps=warps):
            dp.zero_()
            mapped_backward_atoms_factored[(stations * (64 // bn), 64 // bk)](
                dw,
                x,
                dy,
                packed,
                circle,
                section,
                offsets,
                dp,
                M=128,
                N=n,
                K=n,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                BM=16,
                BN=bn,
                BK=bk,
                BA=ba,
                STAGED=True,
                STATION_START=0,
                ROW_START=0,
                num_warps=warps,
                enable_fp_fusion=True,
            )

        try:
            elapsed = milliseconds(run)
            if reference is None:
                reference = dp.clone()
            variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "warps": warps,
                    "ms": elapsed,
                    "max_abs_diff": float((dp - reference).abs().max().item()),
                }
            )
        except Exception as error:  # noqa: BLE001 - report variants independently
            variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "warps": warps,
                    "error": str(error)[:500],
                }
            )
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "window_rows": rows,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
