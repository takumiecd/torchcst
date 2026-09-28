"""Measure the two GPU-heavy pieces of the bounded mapped backward."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_backward import (
    mapped_backward_atoms,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(function, repeats=5):
    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n, m = args.size, args.rows
    assert n % 64 == 0 and m > 0
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
    )
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, n, device="cuda")
    dx = torch.zeros_like(x)
    dp = torch.zeros_like(packed)
    chunk = min(1024, n)
    w = x.new_empty((chunk, n))
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    g = layer.strip.chart.tile_count

    def input_gradient():
        dx.zero_()
        for row_start in range(0, n, chunk):
            rows = min(chunk, n - row_start)
            materialize_logical[(math.ceil(rows / 64), layer.column_groups, 2)](
                packed,
                circle,
                section,
                offsets,
                w,
                N=n,
                K=n,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=g,
                D=4,
                PROFILE=profile,
                BN=64,
                BK=32,
                BA=1,
                FACTORED=True,
                ROW_GROUP_START=row_start // 64,
                LOCAL_W=True,
                num_warps=4,
                enable_fp_fusion=True,
            )
            dx.addmm_(dy[:, row_start : row_start + rows], w[:rows])

    def atom_gradient(bm=16, bn=16, bk=16, ba=8):
        dp.zero_()
        mapped_backward_atoms[(g * (64 // bn), 64 // bk)](
            x,
            dy,
            packed,
            circle,
            section,
            offsets,
            dp,
            M=m,
            N=n,
            K=n,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=g,
            PROFILE=profile,
            BM=bm,
            BN=bn,
            BK=bk,
            BA=ba,
            num_warps=4,
            enable_fp_fusion=False,
        )

    results = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "atoms": packed.shape[0],
        "input_gradient_ms": milliseconds(input_gradient),
        "atom_gradient_baseline_ms": milliseconds(atom_gradient),
    }
    baseline = dp.clone()
    variants = []
    for bn, bk, ba in (
        (16, 16, 1),
        (16, 16, 2),
        (16, 16, 4),
        (16, 16, 8),
        (16, 16, 16),
        (16, 32, 4),
        (16, 32, 8),
        (32, 16, 4),
        (32, 16, 8),
    ):
        try:
            run = lambda bn=bn, bk=bk, ba=ba: atom_gradient(bn=bn, bk=bk, ba=ba)
            elapsed = milliseconds(run, repeats=3)
            difference = (dp - baseline).abs().max().item()
            variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "ms": elapsed,
                    "max_abs_diff": difference,
                }
            )
        except Exception as error:  # noqa: BLE001 - keep later variants running
            variants.append({"bn": bn, "bk": bk, "ba": ba, "error": str(error)[:300]})
    results["variants"] = variants
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
