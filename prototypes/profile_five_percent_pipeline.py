"""Separate A100 five-percent atom preparation and forward kernel costs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from prototypes.block_fused_config import default_fused_config, launch_fused
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    assert args.size in (4096, 8192)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    assert "A100" in props.name
    count = round(args.size * args.size * 0.05)
    layer = BlockStripLinear((args.size, args.size), (64, 64), count, device="cuda")
    x = torch.randn(128, args.size, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    config = default_fused_config(
        layer.tile_shape,
        x.shape[0],
        atom_density=count / (args.size * args.size),
        gpu_name=props.name,
        logical_shape=layer.shape,
    )
    y = torch.empty((128, args.size), device="cuda")
    compiled = launch_fused(layer, x, prepared, y, config)
    view = SimpleNamespace(out_features=args.size, atoms=layer.strip.atoms)
    functions = {
        "prepare": lambda: prepare(
            layer.strip, layer.strip.atoms.p, support_layout=True
        ),
        "prepared_forward": lambda: layer(x, backend="triton_fused", prepared=prepared),
        "full_forward": lambda: layer(x, backend="triton_fused"),
    }
    profiles = [
        capture(view, x, name, fn, False, args.output_dir)
        for name, fn in functions.items()
    ]
    result = {
        "source_commit": args.source_commit,
        "device": props.name,
        "multiprocessors": props.multi_processor_count,
        "shape": [128, args.size, args.size],
        "atoms": count,
        "config": config.__dict__,
        "compiled": {
            "registers_per_thread": compiled.n_regs,
            "compiler_spills": compiled.n_spills,
            "shared_bytes_per_block": compiled.metadata.shared,
        },
        "profiles": profiles,
        "completed": True,
    }
    (args.output_dir / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "compiled": result["compiled"]}))


if __name__ == "__main__":
    main()
