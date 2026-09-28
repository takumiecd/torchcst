"""Time local CST weight generation and forward GEMM separately."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.bounded_gemm import bounded_gemm
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    p, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    x = torch.randn(m, n, device="cuda")
    y = torch.empty_like(x)
    w = torch.empty((1024, n), device="cuda")
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]

    def measure(mode):
        events = {"weight": [], "gemm": []}
        for start in range(0, n, 1024):
            rows = min(1024, n - start)

            def record(name, fn):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                fn()
                end.record()
                events[name].append((begin, end))

            record(
                "weight",
                lambda start=start, rows=rows: materialize_logical[
                    (math.ceil(rows / 64), layer.column_groups, 2)
                ](
                    p,
                    circle,
                    section,
                    offsets,
                    w,
                    N=n,
                    K=n,
                    S=64,
                    T=64,
                    CG=layer.column_groups,
                    G=layer.strip.chart.tile_count,
                    D=4,
                    PROFILE=profile,
                    BN=64,
                    BK=32,
                    BA=1,
                    FACTORED=True,
                    ROW_GROUP_START=start // 64,
                    LOCAL_W=True,
                    num_warps=4,
                    enable_fp_fusion=True,
                ),
            )
            output = y[:, start : start + rows]
            if mode == "tf32x3":
                record(
                    "gemm",
                    lambda output=output, rows=rows: bounded_gemm(
                        x, w[:rows].T, output
                    ),
                )
            else:
                record(
                    "gemm",
                    lambda output=output, rows=rows: torch.mm(
                        x, w[:rows].T, out=output
                    ),
                )
        torch.cuda.synchronize()
        return {
            name: sum(begin.elapsed_time(end) for begin, end in pairs)
            for name, pairs in events.items()
        }

    results = {}
    for mode in ("ieee", "tf32x3"):
        measure(mode)
        samples = [measure(mode) for _ in range(5)]
        results[mode] = {
            key: statistics.median(sample[key] for sample in samples)
            for key in ("weight", "gemm")
        }
    result = {"size": n, "rows": m, "gpu_median_ms": results}
    print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
