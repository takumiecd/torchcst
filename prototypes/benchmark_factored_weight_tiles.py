"""Sweep materialization tile shapes after column-wise distance factoring."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    size = args.size
    assert size in (4096, 8192)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    layer = BlockStripLinear(
        (size, size), (64, 64), round(size * size * 0.05), device="cuda"
    )
    packed = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    expected_w, canonical = mapped_control(layer, packed, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(128, size, device="cuda")
    expected = F.linear(x, expected_w)
    w = torch.empty_like(expected_w)
    options = (
        (16, 32, 4),
        (32, 32, 4),
        (64, 32, 4),
        (32, 64, 4),
        (64, 64, 4),
        (64, 64, 8),
        (32, 16, 4),
        (16, 64, 4),
    )

    def generate(bn, bk, warps):
        p, circle, section, offsets = packed
        return materialize_logical[
            (
                layer.row_groups * math.ceil(64 / bn),
                layer.column_groups,
                math.ceil(64 / bk),
            )
        ](
            p,
            circle,
            section,
            offsets,
            w,
            w,
            N=size,
            K=size,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BN=bn,
            BK=bk,
            BA=1,
            FACTORED=True,
            num_warps=warps,
            enable_fp_fusion=True,
        )

    def generated(bn, bk, warps):
        generate(bn, bk, warps)
        return F.linear(x, w)

    functions = {}
    checks = {}
    resources = {}
    for bn, bk, warps in options:
        name = f"bn{bn}_bk{bk}_w{warps}"
        compiled = generate(bn, bk, warps)
        checks[name] = check(w, expected_w)
        assert checks[name]["passed"], (name, checks[name])
        resources[name] = {
            "registers_per_thread": compiled.n_regs,
            "compiler_spills": compiled.n_spills,
            "shared_bytes_per_block": compiled.metadata.shared,
        }
        functions[name] = lambda bn=bn, bk=bk, warps=warps: generated(bn, bk, warps)
    output_checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert all(value["passed"] for value in output_checks.values())
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [128, size, size],
        "atoms": layer.strip.atoms.p.shape[0],
        "canonical": canonical,
        "weight_checks": checks,
        "output_checks": output_checks,
        "resources": resources,
        "timing": "CUDA Graph 3 rounds rep=20ms; prepared layout; generation plus cuBLAS",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
