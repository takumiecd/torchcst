"""Compare prefix-moment construction with direct atom gradients on 64 stations."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_moment_backward import mapped_backward_atoms_moments
from prototypes.block_prefix_build import build_backward_prefix_moments
from prototypes.block_streamed_backward import (
    mapped_backward_atoms_factored,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(function, repeats=5):
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
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    rows = min(1024, n // 2)
    station_count = min(64, rows // 64 * layer.column_groups)
    x = torch.randn(128, n, device="cuda")
    dy = torch.randn(128, n, device="cuda")
    dw = torch.mm(dy[:, :rows].T, x)
    dp = torch.zeros_like(packed)

    def baseline():
        dp.zero_()
        mapped_backward_atoms_factored[(station_count * 4, 2)](
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
            BN=16,
            BK=32,
            BA=8,
            STAGED=True,
            STATION_START=0,
            ROW_START=0,
            num_warps=4,
            enable_fp_fusion=True,
        )

    direct_ms = milliseconds(baseline)
    reference = dp.clone()
    variants = []
    for bk in (4, 8, 16):
        scratch = torch.empty(
            (station_count, 64 // bk, 20, 64, bk), device="cuda", dtype=torch.float32
        )

        def run(bk=bk, scratch=scratch):
            build_backward_prefix_moments[(station_count, 64 // bk)](
                dw,
                circle,
                section,
                scratch,
                K=n,
                CG=layer.column_groups,
                BK=bk,
                STATION_START=0,
                ROW_START=0,
                num_warps=4,
                enable_fp_fusion=False,
            )

        try:
            elapsed = milliseconds(run)

            def contract(bk=bk, scratch=scratch):
                dp.zero_()
                mapped_backward_atoms_moments[(station_count, 64 // bk)](
                    dw,
                    packed,
                    circle,
                    section,
                    offsets,
                    dp,
                    K=n,
                    CG=layer.column_groups,
                    G=layer.strip.chart.tile_count,
                    BK=bk,
                    STATION_START=0,
                    ROW_START=0,
                    USE_SCRATCH=True,
                    Scratch=scratch,
                    num_warps=4,
                    enable_fp_fusion=False,
                )

            contract_ms = milliseconds(contract)
            difference = (dp - reference).abs()
            variants.append(
                {
                    "bk": bk,
                    "prefix_ms": elapsed,
                    "contract_ms": contract_ms,
                    "scratch_bytes": scratch.numel() * scratch.element_size(),
                    "finite": bool(torch.isfinite(scratch).all().item()),
                    "max_abs_diff": float(difference.max().item()),
                    "relative_l1": float(
                        difference.sum().item() / reference.abs().sum().item()
                    ),
                }
            )
        except Exception as error:  # noqa: BLE001 - report variants independently
            variants.append({"bk": bk, "error": str(error)[:500]})
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "station_count": station_count,
        "direct_atom_gradient_ms": direct_ms,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
