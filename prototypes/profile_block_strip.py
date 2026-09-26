"""Kernel attribution after increasing the number of bounded Strip blocks."""

import argparse
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import execution_plan
from torchcst.nn._backends._triton_preparation_kernels import owners


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "source_commit": args.source_commit,
        "shape": [4096, 4096],
        "batch": 128,
        "atoms": 8192,
        "cases": [],
    }
    for tile in ((64, 64), (128, 128)):
        torch.manual_seed(21)
        layer = BlockStripLinear((4096, 4096), tile, 8192, device="cuda")
        x = torch.randn(128, 4096, device="cuda")
        # capture's forward-only interface needs the logical output size.
        view = SimpleNamespace(out_features=4096, atoms=layer.strip.atoms)
        case = {"tile_shape": tile, "profiles": []}
        for backend in ("triton_fused", "triton_direct"):
            name = f"block_{tile[0]}x{tile[1]}_{backend}"
            prof = capture(
                view, x, name, partial(layer, x, backend=backend), False, args.output_dir
            )
            case["profiles"].append(prof)
        routing = execution_plan(layer.strip).routing
        decoded = layer.strip.chart.geometry.decode_centers(layer.strip.atoms.p[:, 2:])
        a, g = decoded.shape[0], layer.strip.chart.tile_count
        bg = triton.next_power_of_2(g)
        ba = min(32, max(1, 1024 // bg))
        scratch = torch.empty(a, device="cuda", dtype=torch.long)
        compiled = owners[(triton.cdiv(a, ba),)](
            decoded,
            routing.major_radius,
            routing.period,
            routing.starts,
            routing.spans,
            routing.spacing,
            routing.last_row,
            scratch,
            a,
            g,
            *decoded.stride(),
            ba,
            bg,
            enable_fp_fusion=False,
        )
        case["owner_resources"] = {
            "registers_per_thread": compiled.n_regs,
            "compiler_spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared,
            "atom_station_pairs": a * g,
        }
        result["cases"].append(case)
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"tile": tile, "top": [p["kernels"][:2] for p in case["profiles"]]}
            ),
            flush=True,
        )
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
