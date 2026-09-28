"""Sweep bounded backward weight materialization launch configurations."""

import argparse
import json
import math
from pathlib import Path

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_listed_tuning import milliseconds
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    g = layer.strip.chart.tile_count
    w = torch.empty((1024, n), device="cuda")
    reference = None
    results = []
    configs = (
        (64, 32, 4),
        (64, 32, 1),
        (64, 32, 2),
        (64, 32, 8),
        (32, 32, 1),
        (32, 32, 2),
        (32, 32, 4),
        (64, 64, 1),
        (64, 64, 2),
        (64, 64, 4),
        (32, 64, 1),
        (32, 64, 2),
        (32, 64, 4),
    )
    for bn, bk, warps in configs:

        def run(bn=bn, bk=bk, warps=warps):
            return materialize_logical[
                (math.ceil(1024 / bn), layer.column_groups, math.ceil(64 / bk))
            ](
                packed,
                circle,
                section,
                offsets,
                w,
                N=n,
                K=n,
                S=64,
                T=64,
                CG=layer.column_groups,
                G=g,
                D=4,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                BN=bn,
                BK=bk,
                BA=1,
                FACTORED=True,
                ROW_GROUP_START=0,
                LOCAL_W=True,
                num_warps=warps,
                enable_fp_fusion=True,
            )

        try:
            kernel = run()
            torch.cuda.synchronize()
            if reference is None:
                reference = w.clone()
            difference = float((w - reference).abs().max().item())
            ms = milliseconds(run)
            row = {
                "bn": bn,
                "bk": bk,
                "warps": warps,
                "ms": ms,
                "max_abs_diff": difference,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
            }
        except Exception as exc:  # noqa: BLE001 - retain valid variants
            row = {"bn": bn, "bk": bk, "warps": warps, "error": str(exc)[:300]}
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"size": n, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
