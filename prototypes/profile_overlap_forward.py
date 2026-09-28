"""Test overlap of listed CST weight generation with Tensor Core GEMM."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import weight_fp_fusion_enabled
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.bounded_gemm import bounded_gemm
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    atom_lists, list_counts, max_candidates = build_listed_forward_candidates(
        layer, packed, circle, section, offsets
    )
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    x = torch.randn(m, n, device="cuda")
    y = torch.empty_like(x)
    buffers = [torch.empty((1024, n), device="cuda") for _ in range(2)]

    def materialize(buffer, start):
        materialize_listed[(1024 // 64 * layer.column_groups * 4,)](
            packed,
            circle,
            section,
            atom_lists,
            list_counts,
            offsets,
            buffer,
            K=n,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            PROFILE=profile,
            MAX_CANDIDATES=max_candidates,
            STATION_START=start // 64 * layer.column_groups,
            ROW_START=start,
            num_warps=1,
            enable_fp_fusion=weight_fp_fusion_enabled(buffer.device),
        )

    def sequential():
        for start in range(0, n, 1024):
            materialize(buffers[0], start)
            bounded_gemm(x, buffers[0].T, y[:, start : start + 1024])

    default_stream = torch.cuda.current_stream()
    weight_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()

    def pipelined():
        weight_stream.wait_stream(default_stream)
        gemm_stream.wait_stream(default_stream)
        free_events = []
        for i, start in enumerate(range(0, n, 1024)):
            if i >= 2:
                weight_stream.wait_event(free_events[i - 2])
            buffer = buffers[i % 2]
            with torch.cuda.stream(weight_stream):
                materialize(buffer, start)
                ready = torch.cuda.Event()
                ready.record(weight_stream)
            gemm_stream.wait_event(ready)
            with torch.cuda.stream(gemm_stream):
                bounded_gemm(x, buffer.T, y[:, start : start + 1024])
                free = torch.cuda.Event()
                free.record(gemm_stream)
            free_events.append(free)
        default_stream.wait_stream(gemm_stream)

    def measure(fn):
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    sequential_ms = measure(sequential)
    reference = y.clone()
    pipelined_ms = measure(pipelined)
    difference = (y - reference).abs()
    result = {
        "size": n,
        "rows": m,
        "sequential_ms": sequential_ms,
        "pipelined_ms": pipelined_ms,
        "extra_window_bytes": buffers[1].numel() * buffers[1].element_size(),
        "output_max_abs": difference.max().item(),
    }
    print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
