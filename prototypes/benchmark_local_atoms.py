"""Local atom prototype versus I/B fused CST and stored dense on A100."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from prototypes.benchmark_triton_linear import model
from prototypes.local_atom_linear import LocalAtom, forward
from torchcst import (
    CSTLinear,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    StripChart,
    TorusGeometry,
    Triweight,
)
from torchcst.nn._backends._preparation import prepare
from torchcst.nn._backends._triton import forward as fused


def shared_model():
    chart = StripChart(
        shape=(64, 128),
        tile_shape=(16, 128),
        axes=(LinePattern(64, spacing=0.1), GridPattern((8, 16), spacing=0.05)),
        axis=0,
        tile_pitch=2.6,
        geometry=TorusGeometry(
            3,
            major_radius=4 * 2.6 / (2 * math.pi),
            minor_radius=0.4,
            representation="intrinsic",
        ),
    )
    kernel = DirectAmpWidth(
        amplitude_max=1,
        sigma_min=0.8,
        sigma_birth=0.8,
        sigma_max=0.8,
        w_c=0.05,
        profile=Triweight(0.8, normalize_columns=False),
        checkpoint_blocks=False,
    )
    layer = CSTLinear(
        chart=chart, atoms=64, kernel=kernel, device="cuda", backend="triton"
    )
    with torch.no_grad():
        a = torch.arange(64, device="cuda")
        coord = layer.atoms.p.new_zeros((64, 3))
        coord[:, 0] = (
            layer.chart.axes[0].start[0]
            + (a // 16) * 2.6
            + torch.where(a % 2 == 0, 0.7, 2.05)
        )
        layer.atoms.p[:, 2:] = chart.geometry.encode_centers(
            chart.geometry.lift_chart_coordinates(coord)
        )
        layer.atoms.p[:, 0].uniform_(-0.05, 0.05)
    return layer


def probe(batch, atoms, rows, columns, shared=False):
    torch.manual_seed(21)
    layer = shared_model() if shared else model(atoms, rows=rows, columns=columns)
    x = torch.randn(batch, columns, device="cuda", requires_grad=True)
    p = layer.atoms.p
    expected = F.linear(x, layer.dense_weight())
    gradient = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, p), gradient)
    errors = {}
    for bm in (4, 16):
        actual = forward(layer, x, p, batch_tile=bm)
        got = torch.autograd.grad(actual, (x, p), gradient)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got[0], wanted[0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got[1], wanted[1], atol=1e-4, rtol=1e-4)
        errors[bm] = [
            float((a - b).abs().max())
            for a, b in zip((actual, *got), (expected, *wanted), strict=True)
        ]
    del actual, got, expected, wanted
    with torch.no_grad():
        weight = layer.dense_weight().detach()
        packed, circle, section, offsets = prepare(layer, p, support_layout=True)
        bounds = torch.stack((section.amin(0), section.amax(0)))
        counts = torch.diff(offsets).cpu().tolist()
        if shared:
            assert sum(counts[1:-1:2]) > 0
        functions = {
            "dense": lambda: F.linear(x, weight),
            "fused_bm16": lambda: fused(layer, x, p),
            "fused_bm64": lambda: fused(layer, x, p, batch_tile=64),
        }
        for bm in (4, 16):
            functions[f"direct_full_bm{bm}"] = lambda bm=bm: forward(
                layer, x, p, batch_tile=bm
            )
            functions[f"direct_prepared_bm{bm}"] = lambda bm=bm: LocalAtom.apply(
                x, packed, circle, section, offsets, bounds, 16, 1, bm, 128
            )
        samples = {name: [] for name in functions}
        for round_index in range(3):
            names = (
                list(functions) if round_index % 2 == 0 else list(reversed(functions))
            )
            for name in names:
                samples[name].append(do_bench_cudagraph(functions[name], rep=20))
        peaks = {}
        for name in ("fused_bm16", "direct_full_bm4", "direct_full_bm16"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = functions[name]()
            torch.cuda.synchronize()
            peaks[name] = torch.cuda.max_memory_allocated() - base
            del out
    return {
        "shape": [rows, columns],
        "batch": batch,
        "atoms": atoms,
        "shared": shared,
        "bucket_counts": counts,
        "errors": errors,
        "graph_ms": {name: median(v) for name, v in samples.items()},
        "samples": samples,
        "peak_extra_bytes": peaks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--shared-only", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_commit": args.source_commit,
        "cases": [],
    }
    cases = (
        (16, 64, 64, 128),
        (128, 64, 64, 128),
        (16, 256, 64, 128),
        (128, 256, 64, 128),
        (32, 512, 256, 512),
        (128, 512, 256, 512),
        (16, 64, 64, 128, True),
    )
    for case in cases[-1:] if args.shared_only else cases:
        result["cases"].append(probe(*case))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
