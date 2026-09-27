"""Diagnostic: generate transient logical W, then use vendor GEMM at 5%."""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_kernels import _weight_columns_factored, _weight_lanes
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


@triton.jit
def materialize_logical(
    P,
    Circle,
    Section,
    Offsets,
    W,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    FACTORED: tl.constexpr = False,
):
    tile = tl.program_id(0)
    row_group = tile // triton.cdiv(S, BN)
    local = (tile % triton.cdiv(S, BN)) * BN
    column_group = tl.program_id(1)
    start = tl.program_id(2) * BK
    station = row_group * CG + column_group
    if FACTORED:
        tl.static_assert(D == 4 and BA == 1)
        w = _weight_columns_factored(
            P, Circle, Section, Offsets, station, station * S + local, start,
            G * S, T, G, S, PROFILE, BN, BK,
        )
    else:
        w = _weight_lanes(
            P, Circle, Section, Offsets, station, station * S + local, start,
            G * S, T, D, G, S, PROFILE, BN, BK, BA,
        )
    n = row_group * S + local + tl.arange(0, BN)
    k = column_group * T + start + tl.arange(0, BK)
    tl.store(
        W + n[:, None] * K + k[None, :],
        w,
        (n[:, None] < N)
        & (local + tl.arange(0, BN)[:, None] < S)
        & (k[None, :] < K)
        & (start + tl.arange(0, BK)[None, :] < T),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    assert args.size % 64 == 0 and args.batch == 128
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    atoms = round(args.size * args.size * 0.05)
    layer = BlockStripLinear((args.size, args.size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    expected_w, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(args.batch, args.size, device="cuda")
    expected = F.linear(x, expected_w)
    generated_w = torch.empty_like(expected_w)
    factored_w = torch.empty_like(expected_w)
    s, t = layer.tile_shape

    def generate(packed, *, factored=False):
        p, circle, section, offsets = packed
        return materialize_logical[
            (
                layer.row_groups * math.ceil(s / (32 if factored else 16)),
                layer.column_groups,
                math.ceil(t / 32),
            )
        ](
            p,
            circle,
            section,
            offsets,
            factored_w if factored else generated_w,
            N=args.size,
            K=args.size,
            S=s,
            T=t,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=p.shape[1] - 2,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BN=32 if factored else 16,
            BK=32,
            BA=1,
            FACTORED=factored,
            num_warps=4,
            enable_fp_fusion=True,
        )

    def generated(prepared_only, *, factored=False):
        current = (
            prepared
            if prepared_only
            else prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
        )
        generate(current, factored=factored)
        return F.linear(x, factored_w if factored else generated_w)

    generate(prepared)
    generate(prepared, factored=True)
    weight_check = check(generated_w, expected_w)
    factored_weight_check = check(factored_w, expected_w)
    functions = {
        "dense": partial(F.linear, x, expected_w),
        "fused_default": partial(layer, x, backend="triton_fused"),
        "generate_only": partial(generate, prepared),
        "generated_prepared": partial(generated, True),
        "generated_full": partial(generated, False),
        "factored_generate_only": partial(generate, prepared, factored=True),
        "factored_generated_prepared": partial(generated, True, factored=True),
        "factored_generated_full": partial(generated, False, factored=True),
    }
    checks = {
        name: check(fn(), expected)
        for name, fn in functions.items()
        if not name.endswith("generate_only")
    }
    assert weight_check["passed"] and factored_weight_check["passed"]
    assert all(c["passed"] for c in checks.values())
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [args.batch, args.size, args.size],
        "atoms": atoms,
        "canonical": canonical,
        "weight_check": weight_check,
        "factored_weight_check": factored_weight_check,
        "checks": checks,
        "transient_weight_bytes": generated_w.numel() * generated_w.element_size(),
        "timing": "CUDA Graph 3 rounds rep=20ms; dense W precomputed",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
