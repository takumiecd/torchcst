"""Measure bounded-memory factored W generation followed by GEMM at 5% density."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_materialize_five_percent import materialize_logical
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
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
    buffers = {
        chunk: (torch.empty((chunk, size), device="cuda"), torch.empty_like(x))
        for chunk in (64, 128, 256, 512, 1024, size)
    }

    def streamed(chunk, packed):
        p, circle, section, offsets = packed
        w, y = buffers[chunk]
        for start in range(0, size, chunk):
            materialize_logical[(chunk // 32, size // 64, 2)](
                p,
                circle,
                section,
                offsets,
                w,
                N=size,
                K=size,
                S=64,
                T=64,
                CG=size // 64,
                G=layer.strip.chart.tile_count,
                D=p.shape[1] - 2,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                BN=32,
                BK=32,
                BA=1,
                FACTORED=True,
                ROW_GROUP_START=start // 64,
                LOCAL_W=True,
                num_warps=4,
                enable_fp_fusion=True,
            )
            torch.mm(x, w.T, out=y[:, start : start + chunk])
        return y

    def streamed_full(chunk):
        return streamed(
            chunk, prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
        )

    functions = {
        "fused_default": partial(layer, x, backend="triton_fused"),
        **{f"stream_{chunk}": partial(streamed, chunk, prepared) for chunk in buffers},
        **{
            f"stream_{chunk}_full": partial(streamed_full, chunk)
            for chunk in (512, 1024, size)
        },
    }
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert all(result["passed"] for result in checks.values()), checks
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [batch, size, size],
        "atoms": atoms,
        "canonical": canonical,
        "checks": checks,
        "transient_weight_bytes": {
            str(chunk): w.numel() * w.element_size()
            for chunk, (w, _) in buffers.items()
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
