"""Sweep bounded candidate-list construction on one Ada-sized CST layer."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_tile_atom_lists import build_tile_atom_lists
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    capacity = 192
    variants = {}
    for ba in (2, 4, 8, 16, 32, 64, 128):
        for warps in (1, 2, 4, 8):
            name = f"ba{ba}-w{warps}"
            lists = torch.empty((g, 4, capacity), device="cuda", dtype=torch.uint8)
            counts = torch.empty((g, 4), device="cuda", dtype=torch.int32)

            def run(ba=ba, warps=warps, lists=lists, counts=counts):
                build_tile_atom_lists[(g, 4)](
                    packed,
                    circle,
                    section,
                    offsets,
                    lists,
                    counts,
                    G=g,
                    MAX_CANDIDATES=capacity,
                    BA=ba,
                    COMPACT=True,
                    BR=16,
                    BC=64,
                    BOUNDED=True,
                    RANK_LIMIT=256,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )

            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            variants[name] = (lists, counts, graph)

    reference_lists, reference_counts, _ = variants["ba8-w4"]
    reference_lists = reference_lists.clone()
    reference_counts = reference_counts.clone()
    valid = (
        torch.arange(capacity, device="cuda")[None, None, :]
        < reference_counts.clamp_min(0)[:, :, None]
    )
    checks = {}
    for name, (lists, counts, _) in variants.items():
        checks[name] = {
            "count_mismatches": int((counts != reference_counts).sum().item()),
            "list_mismatches": int(((lists != reference_lists) & valid).sum().item()),
        }

    samples = {name: [] for name in variants}
    names = tuple(variants)
    for round_index in range(args.rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            variants[name][2].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rounds": args.rounds,
        "baseline": "ba8-w4",
        "median_ms": {
            name: statistics.median(values) for name, values in samples.items()
        },
        "checks": checks,
        "overflow_tiles": int((reference_counts < 0).sum().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
