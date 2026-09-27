"""Same-run A100 test of fixed site-box culling inside weight generation."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_forward import streamed_forward
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    boxed_prepare,
    station_column_boxes,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    support_boxes = station_site_boxes(plan.circle, plan.section, 64)
    weight_boxes = station_column_boxes(plan.circle, plan.section, 64, 32)
    packed = boxed_prepare(site, site.atoms.p, boxes=support_boxes)
    ordinary = prepare(site, site.atoms.p, support_layout=True)
    assert torch.equal(packed[0], ordinary[0])
    assert torch.equal(packed[3], ordinary[3])
    expected_w, canonical = mapped_control(layer, ordinary, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(128, n, device="cuda")
    expected = F.linear(x, expected_w)
    w = torch.empty_like(expected_w)

    def generate(boxed):
        p, circle, section, offsets = packed
        return materialize_logical[(n // 64, layer.column_groups, 2)](
            p,
            circle,
            section,
            offsets,
            w,
            weight_boxes if boxed else w,
            N=n,
            K=n,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=site.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(site.kernel.profile)],
            BN=64,
            BK=32,
            BA=1,
            FACTORED=True,
            BOX_CULL=boxed,
            num_warps=4,
            enable_fp_fusion=True,
        )

    generate(False)
    regular_weight_check = check(w, expected_w)
    assert regular_weight_check["passed"]
    generate(True)
    boxed_weight_check = check(w, expected_w)
    assert boxed_weight_check["passed"]

    def regular_full():
        return streamed_forward(layer, x, prepared=packed)

    def boxed_full():
        return streamed_forward(layer, x, prepared=packed, materialize_boxes=weight_boxes)

    functions = {
        "generate_regular": partial(generate, False),
        "generate_boxed": partial(generate, True),
        "full_regular": regular_full,
        "full_boxed": boxed_full,
    }
    checks = {
        "regular_weight": regular_weight_check,
        "boxed_weight": boxed_weight_check,
        "full_regular": check(regular_full(), expected),
        "full_boxed": check(boxed_full(), expected),
    }
    assert all(value["passed"] for value in checks.values())
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [128, n, n],
        "atoms": site.atoms.p.shape[0],
        "canonical": canonical,
        "checks": checks,
        "static_weight_box_bytes": weight_boxes.numel() * weight_boxes.element_size(),
        "timing": "CUDA Graph 3 rounds rep=20ms; both paths use boxed support preparation; static boxes built once",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
