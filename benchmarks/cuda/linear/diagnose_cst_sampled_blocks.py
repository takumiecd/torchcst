"""Measure how well fixed geometry-aware anchor samples reconstruct CST W."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.cuda.linear.benchmark_tile_study import mapped_control
from experiments.cuda.linear.anchor_basis import (
    column_basis,
    evenly_spaced_indices,
    interpolation_matrix,
)
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare


def metrics(actual, reference):
    diff = actual - reference
    return {
        "max_abs": float(diff.abs().max()),
        "relative_l2": float(diff.norm() / reference.norm().clamp_min(1e-30)),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    weight, canonical = mapped_control(
        layer,
        prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
        canonical_chunk=4,
    )
    assert canonical["passed"], canonical
    blocks = weight.reshape(16, 64, 16, 64).permute(0, 2, 1, 3).contiguous()
    x = torch.randn((16, n), device="cuda")
    dy = torch.randn((16, n), device="cuda")
    y = x @ weight.T
    dx = dy @ weight
    results = []
    for row_count, col_segment_count in (
        (8, 4),
        (8, 8),
        (12, 8),
        (16, 4),
        (16, 8),
        (16, 12),
        (24, 8),
        (24, 12),
        (32, 8),
    ):
        row_anchors = evenly_spaced_indices(64, row_count)
        u = interpolation_matrix(64, row_anchors).cuda()
        v_cpu, column_anchors = column_basis(col_segment_count)
        v = v_cpu.cuda()
        samples = blocks.index_select(2, torch.tensor(row_anchors, device="cuda"))
        samples = samples.index_select(3, torch.tensor(column_anchors, device="cuda"))
        approx_blocks = torch.matmul(torch.matmul(u, samples), v.T)
        approx = approx_blocks.permute(0, 2, 1, 3).reshape(n, n)
        result = {
            "row_anchors": row_count,
            "column_anchors": len(column_anchors),
            "samples_per_block": row_count * len(column_anchors),
            "weight": metrics(approx, weight),
            "output_m16": metrics(x @ approx.T, y),
            "input_gradient_m16": metrics(dy @ approx, dx),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    singular = torch.linalg.svdvals(blocks)
    energy = singular.square()
    total = energy.sum(-1)
    ranks = {}
    for rank in (4, 8, 12, 16, 24, 32):
        tail = (energy[..., rank:].sum(-1) / total).sqrt()
        ranks[str(rank)] = {
            "median": float(tail.median()),
            "max": float(tail.max()),
        }
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "canonical": canonical,
        "sampled_interpolation": results,
        "block_svd_tail": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
