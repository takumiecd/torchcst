"""Compare initial CST W with trainable global and block low-rank surrogates."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare


def metrics(actual, reference):
    delta = actual - reference
    return {
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    w, canonical = mapped_control(
        layer,
        prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
        canonical_chunk=4,
    )
    assert canonical["passed"], canonical
    x = torch.randn(16, n, device="cuda")
    dy = torch.randn(16, n, device="cuda")
    y = x @ w.T
    dx = dy @ w
    blocks = w.reshape(16, 64, 16, 64).permute(0, 2, 1, 3).contiguous()
    u, s, vh = torch.linalg.svd(blocks, full_matrices=False)
    result = {"device": torch.cuda.get_device_name(), "block": [], "global": []}
    for rank in (4, 8, 12, 16, 24):
        approx_blocks = (u[..., :rank] * s[..., None, :rank]) @ vh[..., :rank, :]
        approx = approx_blocks.permute(0, 2, 1, 3).reshape(n, n)
        item = {
            "rank": rank,
            "parameters": 256 * 128 * rank,
            "weight": metrics(approx, w),
            "output_m16": metrics(x @ approx.T, y),
            "input_gradient_m16": metrics(dy @ approx, dx),
        }
        result["block"].append(item)
        print("BLOCK", json.dumps(item), flush=True)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    for rank in (64, 128, 256, 384, 512):
        approx = (u[:, :rank] * s[None, :rank]) @ vh[:rank, :]
        item = {
            "rank": rank,
            "parameters": 2048 * rank,
            "weight": metrics(approx, w),
            "output_m16": metrics(x @ approx.T, y),
            "input_gradient_m16": metrics(dy @ approx, dx),
        }
        result["global"].append(item)
        print("GLOBAL", json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
