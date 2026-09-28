"""Measure a dense AdamW step under the same CUDA Graph protocol."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--optimizer-mode", choices=("foreach", "fused"), default="foreach"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    with torch.no_grad():
        dense, canonical = mapped_control(
            layer,
            prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
            canonical_chunk=4,
        )
    assert canonical["passed"]
    weight = torch.nn.Parameter(dense.detach())
    del layer, dense
    x = torch.randn((m, n), device="cuda", requires_grad=True)
    dy = torch.randn((m, n), device="cuda")
    optimizer = torch.optim.AdamW(
        [weight], lr=1e-3, capturable=True, **{args.optimizer_mode: True}
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        F.linear(x, weight).backward(dy)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    result = {
        "shape": [m, n, n],
        "device": torch.cuda.get_device_name(),
        "optimizer_mode": args.optimizer_mode,
        "allocated_after_warmup": torch.cuda.memory_allocated(),
    }
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    result["capture_peak_allocated"] = torch.cuda.max_memory_allocated()
    result["allocated_after_capture"] = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    result["median_graph_replay_ms"] = statistics.median(samples)
    result["steady_replay_peak_allocated"] = torch.cuda.max_memory_allocated()
    result["samples_ms"] = samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
