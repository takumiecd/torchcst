"""Test double-buffer overlap of bounded dW GEMMs and atom reductions."""

import argparse
import json
import statistics
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
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
    x = torch.randn(m, n, device="cuda")
    dy = torch.randn(m, n, device="cuda")
    buffers = [torch.empty((1024, n), device="cuda") for _ in range(2)]
    dp = torch.zeros_like(packed)
    profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
    station_count = layer.strip.chart.tile_count
    column_groups = layer.column_groups

    def reduce(buffer, start):
        mapped_backward_atoms_listed[(1024 // 64 * column_groups * 4, 1)](
            buffer,
            packed,
            circle,
            section,
            atom_lists,
            list_counts,
            offsets,
            dp,
            K=n,
            CG=column_groups,
            G=station_count,
            PROFILE=profile,
            MAX_CANDIDATES=max_candidates,
            BA=1,
            COMPACT=True,
            BR=16,
            BC=64,
            STATION_START=start // 64 * column_groups,
            ROW_START=start,
            OPT_TRIWEIGHT=True,
            num_warps=1,
            enable_fp_fusion=True,
        )

    def sequential():
        dp.zero_()
        for start in range(0, n, 1024):
            torch.mm(dy[:, start : start + 1024].T, x, out=buffers[0])
            reduce(buffers[0], start)

    default_stream = torch.cuda.current_stream()
    mm_stream = torch.cuda.Stream()
    atom_stream = torch.cuda.Stream()

    def pipelined():
        dp.zero_()
        mm_stream.wait_stream(default_stream)
        atom_stream.wait_stream(default_stream)
        free_events = []
        for i, start in enumerate(range(0, n, 1024)):
            if i >= 2:
                mm_stream.wait_event(free_events[i - 2])
            buffer = buffers[i % 2]
            with torch.cuda.stream(mm_stream):
                torch.mm(dy[:, start : start + 1024].T, x, out=buffer)
                ready = torch.cuda.Event()
                ready.record(mm_stream)
            atom_stream.wait_event(ready)
            with torch.cuda.stream(atom_stream):
                reduce(buffer, start)
                free = torch.cuda.Event()
                free.record(atom_stream)
            free_events.append(free)
        default_stream.wait_stream(atom_stream)

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
    reference = dp.clone()
    pipelined_ms = measure(pipelined)
    difference = (dp - reference).abs()
    tolerance = 3e-4 + 3e-4 * reference.abs()
    result = {
        "size": n,
        "rows": m,
        "sequential_ms": sequential_ms,
        "pipelined_ms": pipelined_ms,
        "extra_window_bytes": buffers[1].numel() * buffers[1].element_size(),
        "gradient_max_abs": difference.max().item(),
        "gradient_violations": (difference > tolerance).sum().item(),
    }
    print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
