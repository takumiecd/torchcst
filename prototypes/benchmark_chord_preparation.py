"""Compare exact-neighbor chord search against full 64-row support routing."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_tile_study import timing
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    assert args.size in (4096, 8192)
    props = torch.cuda.get_device_properties(0)
    assert "A100" in props.name
    torch.manual_seed(21)
    count = round(args.size * args.size * 0.05)
    layer = BlockStripLinear((args.size, args.size), (64, 64), count, device="cuda")
    x = torch.randn(128, args.size, device="cuda")

    def baseline_prepare():
        return prepare(layer.strip, layer.strip.atoms.p, support_layout=True)

    def chord_prepare():
        return prepare(
            layer.strip,
            layer.strip.atoms.p,
            support_layout=True,
            support_chord_nearest=True,
        )

    checks = []
    for drift in (0.0, 0.03):
        if drift:
            layer.strip.atoms.p[:, 2] += drift
        baseline = baseline_prepare()
        chord = chord_prepare()
        same = [torch.equal(a, b) for a, b in zip(baseline, chord)]
        checks.append({"drift": drift, "same_tensors": same})
        assert all(same), checks
    functions = {
        "prepare_baseline": baseline_prepare,
        "prepare_chord": chord_prepare,
        "forward_baseline": lambda: layer(x, backend="triton_fused"),
        "forward_chord": lambda: layer(
            x, backend="triton_fused", prepared=chord_prepare()
        ),
    }
    baseline_y, chord_y = functions["forward_baseline"](), functions["forward_chord"]()
    torch.testing.assert_close(baseline_y, chord_y, atol=0, rtol=0)
    result = {
        "source_commit": args.source_commit,
        "device": props.name,
        "multiprocessors": props.multi_processor_count,
        "shape": [128, args.size, args.size],
        "atoms": count,
        "same_prepared": checks,
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "median_ms": result["median_ms"]}))


if __name__ == "__main__":
    main()
