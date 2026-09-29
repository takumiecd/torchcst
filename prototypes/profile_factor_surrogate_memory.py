"""Measure one isolated CUDA Graph training step's allocated peak."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "factor", "cst"), required=True)
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    if args.mode in ("dense", "factor"):
        with torch.no_grad():
            weight, canonical = mapped_control(
                layer,
                prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
                canonical_chunk=4,
            )
            assert canonical["passed"], canonical
            if args.mode == "dense":
                dense_weight = torch.nn.Parameter(weight.clone())
                parameters = [dense_weight]
            else:
                left, singular, right_t = torch.linalg.svd(weight, full_matrices=False)
                scale = singular[: args.rank].sqrt()
                left_factor = torch.nn.Parameter(
                    (left[:, : args.rank] * scale).contiguous()
                )
                right_factor = torch.nn.Parameter(
                    (right_t[: args.rank, :].T * scale).contiguous()
                )
                parameters = [left_factor, right_factor]
                del left, singular, right_t, scale
            del weight
        del layer
    else:
        plan = execution_plan(layer.strip)
        boxes = station_site_boxes(plan.circle, plan.section, 64)
        hints = balanced_home_columns(layer.strip)
        parameters = [layer.strip.atoms.p]
    gc.collect()
    torch.cuda.empty_cache()
    x = torch.randn(args.rows, n, device="cuda", requires_grad=True)
    dy = torch.randn(args.rows, n, device="cuda")
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=True, capturable=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        if args.mode == "dense":
            output = F.linear(x, dense_weight)
        elif args.mode == "factor":
            output = (x @ right_factor) @ left_factor.T
        else:
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=512,
                cache_windows=2,
                cache_weight_dtype=torch.float16,
                weight_tile_rows=8,
                atom_kernel="staged_listed",
                gemm_mode="ieee",
                forward_gemm_mode="ieee",
                materialize_mode="listed_bounded",
                listed_unroll=4,
                listed_builder_ba=32,
                listed_builder_warps=1,
            )
        output.backward(dy)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    after_warmup = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    capture_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "mode": args.mode,
        "rows": args.rows,
        "rank": args.rank if args.mode == "factor" else None,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "allocated_after_warmup": after_warmup,
        "capture_peak_allocated": capture_peak,
        "steady_replay_peak_allocated": torch.cuda.max_memory_allocated(),
        "scope": "mode isolated after one-time dense/SVD initialization; PyTorch allocated bytes",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
