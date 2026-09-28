"""Locate the live-allocation peak in five-percent boxed atom preparation."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    boxed_prepare,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--fast-decode", action="store_true")
    parser.add_argument(
        "--sort-mode", choices=("sort_stable", "argsort_stable", "sort_unstable"),
        default="sort_stable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    x = torch.randn(128, n, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    sort_events = []
    decode_events = []
    original_sort = torch.sort
    geometry_type = type(site.chart.geometry)
    original_decode = geometry_type.decode_centers

    def measured_decode(geometry, centers):
        torch.cuda.synchronize()
        event = {
            "live_before_bytes": torch.cuda.memory_allocated(),
            "peak_before_bytes": torch.cuda.max_memory_allocated(),
        }
        decoded = original_decode(geometry, centers)
        torch.cuda.synchronize()
        event["live_after_bytes"] = torch.cuda.memory_allocated()
        event["peak_after_bytes"] = torch.cuda.max_memory_allocated()
        decode_events.append(event)
        return decoded

    def measured_sort(*sort_args, **sort_kwargs):
        torch.cuda.synchronize()
        event = {
            "live_before_bytes": torch.cuda.memory_allocated(),
            "peak_before_bytes": torch.cuda.max_memory_allocated(),
            "stable": sort_kwargs.get("stable"),
            "input_bytes": sort_args[0].numel() * sort_args[0].element_size(),
        }
        if args.sort_mode == "sort_stable":
            sorted_result = original_sort(*sort_args, **sort_kwargs)
        elif args.sort_mode == "argsort_stable":
            order = torch.argsort(sort_args[0], stable=True)
            sorted_result = sort_args[0][order], order
        else:
            sorted_result = original_sort(sort_args[0], stable=False)
        torch.cuda.synchronize()
        event["live_after_bytes"] = torch.cuda.memory_allocated()
        event["peak_after_bytes"] = torch.cuda.max_memory_allocated()
        sort_events.append(event)
        return sorted_result

    torch.sort = measured_sort
    geometry_type.decode_centers = measured_decode
    try:
        prepared = boxed_prepare(
            site,
            site.atoms.p,
            boxes=boxes,
            witness_cols=hints,
            fast_witness=True,
            fast_decode=args.fast_decode,
        )
    finally:
        torch.sort = original_sort
        geometry_type.decode_centers = original_decode
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "source_commit": args.source_commit,
        "device": torch.cuda.get_device_name(),
        "shape": [128, n, n],
        "atoms": site.atoms.p.shape[0],
        "fast_decode": args.fast_decode,
        "sort_mode": args.sort_mode,
        "input_bytes": x.numel() * x.element_size(),
        "atom_parameter_bytes": site.atoms.p.numel() * site.atoms.p.element_size(),
        "prepared_payload_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in prepared
        ),
        "baseline_live_bytes": before,
        "decode_events": decode_events,
        "sort_events": sort_events,
        "after_prepare_live_bytes": after,
        "prepare_peak_bytes": peak,
        "completed": True,
    }
    assert len(sort_events) == 1, sort_events
    assert len(decode_events) == (0 if args.fast_decode else 1), decode_events
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "completed",
                "decode": decode_events,
                "sort": sort_events,
                "peak": peak,
            }
        )
    )


if __name__ == "__main__":
    main()
