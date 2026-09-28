"""Compare staged site-major and station atom-major gradient kernels."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_atom_major_backward import mapped_backward_atoms_station
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
    )
    rows = min(1024, n // 2)
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, n, device="cuda")
    dw = torch.mm(dy[:, :rows].T, x)
    dp = torch.zeros_like(packed)
    g = layer.strip.chart.tile_count
    stations = rows // 64 * layer.column_groups
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]

    def site_major(bn=16, bk=32, ba=8):
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
            G=g,
            PROFILE=profile,
            BM=16,
            BN=bn,
            BK=bk,
            BA=ba,
            STAGED=True,
            STATION_START=0,
            ROW_START=0,
            num_warps=4,
            enable_fp_fusion=False,
        )

    baseline_ms = milliseconds(site_major)
    reference = dp.clone()
    counts = torch.diff(offsets)
    site_variants = []
    for bn, bk, ba in (
        (8, 32, 8),
        (8, 32, 16),
        (16, 16, 16),
        (16, 32, 4),
        (16, 32, 16),
        (16, 32, 32),
        (16, 64, 4),
        (16, 64, 8),
        (16, 64, 16),
        (32, 16, 8),
        (32, 32, 8),
    ):
        try:
            run = lambda bn=bn, bk=bk, ba=ba: site_major(bn, bk, ba)
            elapsed = milliseconds(run)
            site_variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "ms": elapsed,
                    "max_abs_diff": float((dp - reference).abs().max().item()),
                }
            )
        except Exception as error:  # noqa: BLE001 - report variants independently
            site_variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "error": str(error)[:400],
                }
            )
    variants = []
    for bn, bk, ba, lanes, warps in (
        (8, 16, 16, 16, 4),
        (8, 16, 32, 8, 4),
        (8, 32, 16, 8, 4),
        (8, 32, 16, 16, 4),
        (16, 16, 16, 8, 4),
        (16, 16, 16, 16, 4),
        (16, 32, 2, 16, 4),
        (16, 32, 4, 8, 4),
        (16, 32, 4, 16, 4),
        (16, 32, 4, 32, 4),
        (16, 32, 8, 8, 4),
        (16, 32, 8, 16, 4),
        (16, 32, 8, 32, 4),
        (16, 64, 4, 16, 4),
        (32, 32, 4, 16, 4),
    ):

        def run(bn=bn, bk=bk, ba=ba, lanes=lanes, warps=warps):
            dp.zero_()
            mapped_backward_atoms_station[(stations, lanes)](
                dw,
                packed,
                circle,
                section,
                offsets,
                dp,
                K=n,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=g,
                PROFILE=profile,
                BN=bn,
                BK=bk,
                BA=ba,
                LANES=lanes,
                STATION_START=0,
                ROW_START=0,
                num_warps=warps,
                enable_fp_fusion=False,
            )

        try:
            elapsed = milliseconds(run)
            variants.append(
                {
                    "bn": bn,
                    "bk": bk,
                    "ba": ba,
                    "lanes": lanes,
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
                    "lanes": lanes,
                    "warps": warps,
                    "error": str(error)[:400],
                }
            )
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "atoms": packed.shape[0],
        "window_rows": rows,
        "max_bucket_count": int(counts.max().item()),
        "median_bucket_count": float(counts.median().item()),
        "site_major_ms": baseline_ms,
        "site_variants": site_variants,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
