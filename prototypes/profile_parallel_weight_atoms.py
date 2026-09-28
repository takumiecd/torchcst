"""Compare serial and parallel atom reduction for a bounded W window."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_materialize_parallel import materialize_listed_parallel
from prototypes.block_materialize_subtile import materialize_listed_subtile
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import weight_fp_fusion_enabled
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    window_rows = min(1024, n // 2)
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
    reference = torch.empty((window_rows, n), device="cuda")
    actual = torch.empty_like(reference)
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    grid = (window_rows // 64 * layer.column_groups * 4,)
    common = {
        "K": n,
        "CG": layer.column_groups,
        "G": layer.strip.chart.tile_count,
        "PROFILE": profile,
        "MAX_CANDIDATES": max_candidates,
        "STATION_START": 0,
        "ROW_START": 0,
        "enable_fp_fusion": weight_fp_fusion_enabled(reference.device),
    }

    def baseline():
        return materialize_listed[grid](
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            reference,
            num_warps=1,
            **common,
        )

    def variant(ba, warps):
        return materialize_listed_parallel[grid](
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            actual,
            BA=ba,
            num_warps=warps,
            **common,
        )

    def measure(fn):
        compiled = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(7):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
        return statistics.median(times), times, compiled

    baseline_ms, baseline_times, compiled = measure(baseline)
    x = torch.randn((m, n), device="cuda")
    expected = x @ reference.T
    if args.subtile:
        cases = []
        for br, bc, ba, warps in (
            (4, 16, 16, 4),
            (4, 16, 32, 4),
            (8, 16, 8, 4),
            (8, 16, 16, 4),
        ):
            per_station = 4 * (16 // br) * (64 // bc)

            def subtile(per_station=per_station, br=br, bc=bc, ba=ba, warps=warps):
                return materialize_listed_subtile[
                    (window_rows // 64 * layer.column_groups * per_station,)
                ](
                    packed,
                    circle,
                    section,
                    lists,
                    counts,
                    offsets,
                    actual,
                    BR=br,
                    BC=bc,
                    BA=ba,
                    num_warps=warps,
                    **common,
                )

            try:
                ms, times, kernel = measure(subtile)
                error = (actual - reference).abs()
                output = x @ actual.T
                output_error = (output - expected).abs()
                tolerance = 3e-5 + 3e-5 * expected.abs()
                row = {
                    "br": br,
                    "bc": bc,
                    "ba": ba,
                    "warps": warps,
                    "median_ms": ms,
                    "samples_ms": times,
                    "weight_max_abs": error.max().item(),
                    "output_max_abs": output_error.max().item(),
                    "output_violations": (output_error > tolerance).sum().item(),
                    "registers": kernel.n_regs,
                    "spills": kernel.n_spills,
                }
            except Exception as exc:  # noqa: BLE001 - keep remaining candidates
                row = {"br": br, "bc": bc, "ba": ba, "error": str(exc)[:400]}
            cases.append(row)
            print(json.dumps(row), flush=True)
        result = {
            "device": torch.cuda.get_device_name(),
            "size": n,
            "rows": m,
            "window_rows": window_rows,
            "baseline_ms": baseline_ms,
            "baseline_samples_ms": baseline_times,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    results = []
    for ba in (2, 4, 8):
        for warps in (1, 2, 4):
            try:
                ms, times, kernel = measure(
                    lambda ba=ba, warps=warps: variant(ba, warps)
                )
                error = (actual - reference).abs()
                output = x @ actual.T
                output_error = (output - expected).abs()
                tolerance = 3e-5 + 3e-5 * expected.abs()
                row = {
                    "ba": ba,
                    "warps": warps,
                    "median_ms": ms,
                    "samples_ms": times,
                    "weight_max_abs": error.max().item(),
                    "output_max_abs": output_error.max().item(),
                    "output_violations": (output_error > tolerance).sum().item(),
                    "registers": kernel.n_regs,
                    "spills": kernel.n_spills,
                }
            except Exception as exc:  # noqa: BLE001 - retain other variants
                row = {"ba": ba, "warps": warps, "error": str(exc)[:400]}
            results.append(row)
            print(json.dumps(row), flush=True)
    paired = {"serial": [], "ba2_warps4": []}
    for index in range(12):
        for name, fn in (
            (("serial", baseline), ("ba2_warps4", lambda: variant(2, 4)))
            if index % 2 == 0
            else (("ba2_warps4", lambda: variant(2, 4)), ("serial", baseline))
        ):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            paired[name].append(begin.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "window_rows": window_rows,
        "baseline_ms": baseline_ms,
        "baseline_samples_ms": baseline_times,
        "baseline_registers": compiled.n_regs,
        "baseline_spills": compiled.n_spills,
        "paired_medians_ms": {
            key: statistics.median(value) for key, value in paired.items()
        },
        "paired_samples_ms": paired,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
