"""Measure bounded-memory factored W generation followed by GEMM at 5% density."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    size, batch = args.size, 128
    assert size in (4096, 8192)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    atoms = round(size * size * 0.05)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    expected_w, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(batch, size, device="cuda")
    expected = F.linear(x, expected_w)
    chunks = (64, 128, 256, 512, 1024, size)

    def streamed(chunk, packed):
        return layer(
            x, backend="triton_streamed", prepared=packed, weight_chunk_rows=chunk
        )

    def streamed_full(chunk):
        return layer(x, backend="triton_streamed", weight_chunk_rows=chunk)

    functions = {
        "fused_default": partial(layer, x, backend="triton_fused"),
        **{f"stream_{chunk}": partial(streamed, chunk, prepared) for chunk in chunks},
        **{
            f"stream_{chunk}_full": partial(streamed_full, chunk)
            for chunk in (512, 1024, size)
        },
    }
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert all(result["passed"] for result in checks.values()), checks
    eager = {}
    memory = {}
    if args.eager:
        selected = ("fused_default", "stream_1024_full")
        for name in selected:
            for _ in range(3):
                functions[name]()
        torch.cuda.synchronize()
        samples = {name: [] for name in selected}
        for round_index in range(3):
            for name in selected if round_index % 2 == 0 else reversed(selected):
                fn = functions[name]
                start = time.perf_counter()
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 100)
        eager = {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in samples.items()
        }
        for name in selected:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            result_tensor = functions[name]()
            torch.cuda.synchronize()
            memory[name] = torch.cuda.max_memory_allocated() - before
            del result_tensor
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [batch, size, size],
        "atoms": atoms,
        "canonical": canonical,
        "checks": checks,
        "eager_wall_time": eager,
        "peak_allocated_increment_bytes": memory,
        "transient_weight_bytes": {
            str(chunk): chunk * size * x.element_size() for chunk in chunks
        },
        "timing": "CUDA Graph 3 rounds rep=20ms; _full paths include support preparation",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
