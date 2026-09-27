"""Compare stable polar-chord distance factoring with Cartesian differences."""

import argparse
import json
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
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    expected_w, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(128, size, device="cuda")
    expected = F.linear(x, expected_w)
    w = torch.empty_like(expected_w)

    def generate(polar, bk):
        p, circle, section, offsets = prepared
        return materialize_logical[(layer.row_groups, layer.column_groups, 64 // bk)](
            p,
            circle,
            section,
            offsets,
            w,
            N=size,
            K=size,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BN=64,
            BK=bk,
            BA=1,
            FACTORED=True,
            POLAR_CHORD=polar,
            num_warps=4,
            enable_fp_fusion=True,
        )

    functions = {}
    checks = {}
    resources = {}
    for polar, bk in ((False, 32), (True, 32), (False, 64), (True, 64)):
        name = f"{'polar' if polar else 'cartesian'}_bk{bk}"
        compiled = generate(polar, bk)
        checks[name] = {
            "weight": check(w, expected_w),
            "output": check(F.linear(x, w), expected),
        }
        resources[name] = {
            "registers_per_thread": compiled.n_regs,
            "compiler_spills": compiled.n_spills,
        }

        def generated(polar=polar, bk=bk):
            generate(polar, bk)
            return F.linear(x, w)

        functions[name] = generated
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "shape": [128, size, size],
        "atoms": layer.strip.atoms.p.shape[0],
        "canonical": canonical,
        "checks": checks,
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
