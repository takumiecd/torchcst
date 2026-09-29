"""Compare separate dW GEMM and fused listed atom gradient on 1024²."""

import argparse
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_streamed_backward import (
    trainable_boxed_prepare,
    build_listed_forward_candidates_bounded,
)
from prototypes.block_tile_atom_lists import mapped_backward_atoms_listed
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, PROFILE_KINDS

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(21)
n = 1024
layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
plan = execution_plan(layer.strip)
boxes = station_site_boxes(plan.circle, plan.section, 64)
hints = balanced_home_columns(layer.strip)
p, circle, section, offsets = trainable_boxed_prepare(
    layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
)
lists, counts, cap = build_listed_forward_candidates_bounded(
    layer, p, circle, section, offsets, ba=32, warps=1
)
g = layer.strip.chart.tile_count
cg = layer.column_groups
profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
w = torch.empty((512, n), device="cuda")
dp = torch.zeros_like(p)
results = []
for m in (16, 128):
    x = torch.randn((m, n), device="cuda")
    dy = torch.randn((m, n), device="cuda")

    def grad(fused):
        mapped_backward_atoms_listed[(512 // 64 * cg * 4, 1)](
            w,
            p,
            circle,
            section,
            lists,
            counts,
            offsets,
            dp,
            K=n,
            CG=cg,
            G=g,
            PROFILE=profile,
            MAX_CANDIDATES=cap,
            BOUNDED=True,
            BA=1,
            COMPACT=True,
            BR=16,
            BC=64,
            STATION_START=0,
            ROW_START=0,
            OPT_TRIWEIGHT=True,
            X=x,
            DY=dy,
            M=m,
            N=n,
            FUSED_DW=fused,
            num_warps=1,
            enable_fp_fusion=True,
        )

    def separate():
        torch.mm(dy[:, :512].T, x, out=w)
        grad(False)

    def fused():
        grad(True)

    dp.zero_()
    separate()
    torch.cuda.synchronize()
    ref = dp.clone()
    dp.zero_()
    fused()
    torch.cuda.synchronize()
    delta = float((dp - ref).abs().max())
    allclose = bool(torch.allclose(dp, ref, atol=1e-3, rtol=1e-3))
    for name, fn in [("separate", separate), ("fused", fused)]:
        try:
            ms = do_bench_cudagraph(fn, rep=100)
            r = {
                "rows": m,
                "algorithm": name,
                "us": ms * 1000,
                "max_gradient_delta": delta,
                "gradient_allclose": allclose,
            }
        except Exception as e:
            r = {"rows": m, "algorithm": name, "error": str(e)[:500]}
        results.append(r)
        print(r, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2) + "\n")
