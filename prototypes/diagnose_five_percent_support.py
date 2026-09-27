"""Measure whether column-fragment atom support culling is useful at 5%."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_support_diagnostics import diagnose_support
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 8192])
    args = parser.parse_args()
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        torch.manual_seed(21)
        assert size % 64 == 0
        atoms = round(size * size * 0.05)
        layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
        prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
        diagnostic = diagnose_support(layer, prepared, 128)
        result["cases"].append({"size": size, "atoms": atoms, **diagnostic})
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
