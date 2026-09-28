"""Measure the CPU synchronization cost of sizing listed atom candidates.

The fixed capacity is measured once from immutable offsets before timing, so
this is a diagnostic and is not a safe training-time capacity policy.
"""

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
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    prepared = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    original = build_listed_forward_candidates(layer, *prepared)
    capacity = original[2]
    fixed = build_listed_forward_candidates(
        layer, *prepared, unchecked_fixed_capacity=capacity
    )
    torch.cuda.synchronize()
    assert capacity > 0
    assert torch.equal(original[1], fixed[1])
    valid = torch.arange(capacity, device="cuda") < original[1][..., None]
    assert torch.equal(original[0][valid], fixed[0][valid])

    samples = {"dynamic": [], "fixed": []}
    for i in range(args.rounds):
        for name in ("dynamic", "fixed") if i % 2 == 0 else ("fixed", "dynamic"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            build_listed_forward_candidates(
                layer,
                *prepared,
                unchecked_fixed_capacity=capacity if name == "fixed" else None,
            )
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "capacity": capacity,
        "scope": "candidate-list build from fixed prepared tensors; exact list equality; separate GPU sync on each timed call",
        "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
