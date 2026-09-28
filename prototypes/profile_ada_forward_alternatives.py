"""Compare bounded W+GEMM with mapped fused forward on Ada."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.benchmark_large_forward import check
from prototypes.block_fused_config import FusedConfig, launch_fused
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
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    prepared = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists = build_listed_forward_candidates(layer, *prepared)
    x = torch.randn(m, n, device="cuda")
    y = torch.empty((m, n), device="cuda")
    window_rows = min(1024, n // 2)
    cache_rows = 0 if n == 1024 else 2048

    def bounded():
        return streamed_forward(
            layer,
            x,
            prepared=prepared,
            weight_chunk_rows=window_rows,
            cache_weight_rows=cache_rows,
            materialize_mode="listed",
            listed_data=lists,
            gemm_mode="ieee" if m == 128 or n == 1024 else "tf32x3",
        )

    configs = {
        "a100_fused_shape": FusedConfig(128, 32, 32, 1, 4, True, False, 8, True, True),
        "simple_fused": FusedConfig(128, 16, 16, 8, 4, True, False),
    }
    reference = bounded()

    def measure(fn):
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples), samples

    bounded_ms, bounded_samples = measure(bounded)
    cases = {}
    for name, config in configs.items():

        def run(config=config):
            return launch_fused(layer, x, prepared, y, config)

        try:
            kernel = run()
            torch.cuda.synchronize()
            accuracy = check(y, reference)
            elapsed, samples = measure(run)
            cases[name] = {
                "ms": elapsed,
                "samples_ms": samples,
                "accuracy": accuracy,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "scratch_bytes": config.split_k * m * n * 4
                if config.split_k > 1
                else 0,
            }
        except Exception as error:  # noqa: BLE001 - retain other result
            cases[name] = {"error": str(error)[:500]}
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "size": n,
        "batch_rows": m,
        "bounded_ms": bounded_ms,
        "bounded_samples_ms": bounded_samples,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
