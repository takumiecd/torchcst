"""Measure eager versus CUDA Graph replay for prepared native CST forward."""

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
from prototypes.block_streamed_forward import streamed_forward
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    prepared = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists = build_listed_forward_candidates(layer, *prepared)
    x = torch.randn((m, n), device="cuda")

    def eager():
        return streamed_forward(
            layer,
            x,
            prepared=prepared,
            listed_data=lists,
            weight_chunk_rows=1024,
            cache_weight_rows=2048,
            materialize_mode="listed",
            gemm_mode="ieee" if m == 128 else "tf32x3",
        )

    for _ in range(3):
        reference = eager()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = eager()
    graph.replay()
    torch.cuda.synchronize()
    accuracy = {
        "max_abs": (reference - captured).abs().max().item(),
        "same": bool(torch.equal(reference, captured)),
    }

    functions = {"eager": eager, "graph": graph.replay}
    samples = {name: [] for name in functions}
    for round_index in range(12):
        order = ("eager", "graph") if round_index % 2 == 0 else ("graph", "eager")
        for name in order:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - begin) * 1000)
    cases = {
        name: {
            "median_ms": statistics.median(values[1:]),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "scope": "prepared forward only; preparation, backward, and optimizer excluded",
        "accuracy": accuracy,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
