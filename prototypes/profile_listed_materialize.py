"""Compare listed and unlisted bounded CST weight generation."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import build_tile_atom_lists
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def milliseconds(fn):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--disable-fusion", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    p, circle, section, offsets = trainable_boxed_prepare(
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
    list_dtype = torch.uint8 if max_candidates <= 256 else torch.uint16
    lists = torch.empty((g, 4, max_candidates), device="cuda", dtype=list_dtype)
    counts = torch.empty((g, 4), device="cuda", dtype=torch.int32)
    reference = torch.empty((1024, n), device="cuda")
    actual = torch.empty_like(reference)
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]

    def build():
        build_tile_atom_lists[(g, 4)](
            p,
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

    list_ms = milliseconds(build)
    comparisons = []
    results = {}
    for warps in (1, 2, 4):
        original_ms, listed_ms = 0.0, 0.0
        for start in range(0, n, 1024):
            rows = min(1024, n - start)

            def original(start=start, rows=rows):
                materialize_logical[(rows // 64, layer.column_groups, 2)](
                    p,
                    circle,
                    section,
                    offsets,
                    reference,
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
                    ROW_GROUP_START=start // 64,
                    LOCAL_W=True,
                    num_warps=4,
                    enable_fp_fusion=not args.disable_fusion,
                )

            def listed(start=start, rows=rows, warps=warps):
                materialize_listed[(rows // 64 * layer.column_groups * 4,)](
                    p,
                    circle,
                    section,
                    lists,
                    counts,
                    offsets,
                    actual,
                    K=n,
                    CG=layer.column_groups,
                    G=g,
                    PROFILE=profile,
                    MAX_CANDIDATES=max_candidates,
                    STATION_START=start // 64 * layer.column_groups,
                    ROW_START=start,
                    num_warps=warps,
                    enable_fp_fusion=not args.disable_fusion,
                )

            original_ms += milliseconds(original)
            listed_ms += milliseconds(listed)
            difference = (reference[:rows] - actual[:rows]).abs()
            tolerance = 3e-5 + 3e-5 * reference[:rows].abs()
            comparisons.append(
                {
                    "max_abs": difference.max().item(),
                    "violations": (difference > tolerance).sum().item(),
                }
            )
        results[str(warps)] = {"original_ms": original_ms, "listed_ms": listed_ms}
    result = {
        "size": n,
        "enable_fp_fusion": not args.disable_fusion,
        "max_candidates": max_candidates,
        "list_ms": list_ms,
        "warps": results,
        "max_abs": max(item["max_abs"] for item in comparisons),
        "violations": max(item["violations"] for item in comparisons),
    }
    print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
