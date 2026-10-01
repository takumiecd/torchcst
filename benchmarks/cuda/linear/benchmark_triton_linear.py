"""Reproducible generated-weight GEMM probe; run on a CUDA GPU.

python -m benchmarks.cuda.linear.benchmark_triton_linear --output benchmark.json
Reports preparation, end-to-end training, and fused versus split execution.
The split control uses the exact same Triton weight generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, presets
from torchcst._backends.cuda.algorithms.strip_torus.fused.executor import _FusedLinear
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import (
    prepare as prepare_atoms,
)
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.kernels.execution import KernelOptions
from torchcst.kernels import ProfileBinding


def timed(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def model(atoms, *, rows=64, columns=128, station_rows=16):
    stations = math.ceil(rows / station_rows)
    chart = _construction.strip(
        shape=(rows, columns),
        tile_shape=(station_rows, columns),
        axes=(
            _construction.line_pattern(
                rows, spacing=min(0.1, 1.5 / max(station_rows - 1, 1))
            ),
            _construction.grid_pattern((columns // 16, 16), spacing=0.05),
        ),
        axis=0,
        tile_pitch=4.1,
        geometry=_construction.torus(
            3,
            major_radius=max(2, stations) * 4.1 / (2 * math.pi),
            minor_radius=0.4,
            representation="intrinsic",
        ),
    )
    kernel = presets.direct_activity(
        amplitude_max=1.0,
        input_bounds=BandwidthBounds(
            minimum=0.2, birth=0.5, maximum=0.8, upper_floor=0.2
        ),
        w_c=0.05,
        profile=ProfileBinding(profile=TriweightSpec()),
    )
    return CSTLinear(
        chart=chart,
        atoms=atoms,
        kernel=kernel,
        kernel_options=KernelOptions(checkpoint_blocks=False),
        backend="triton",
        device="cuda",
    )


def prepare(layer):
    return prepare_atoms(layer, layer.atoms.p)


def probe(batch, atoms, repeats):
    from triton.testing import do_bench_cudagraph

    from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import (
        materialize_weights,
    )

    torch.manual_seed(21)
    layer = model(atoms)
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    grad = torch.randn(batch, 64, device="cuda")
    result = {
        "batch": batch,
        "atoms": atoms,
        "shape": [64, 128],
        "tile_shape": [16, 128],
    }
    with torch.no_grad():
        result["prepare_ms"] = timed(lambda: prepare(layer), repeats)
        packed, circle, section, offsets = prepare(layer)
        for backend in ("materialized", "tiled", "triton"):
            layer.backend = backend
            result[backend + "_forward_ms"] = timed(lambda: layer(x), repeats)

        def fused():
            return _FusedLinear.apply(x, packed, circle, section, offsets, 16, 1)

        weight = torch.empty(64, 128, device="cuda")

        def split():
            materialize_weights[(4, 8)](
                packed,
                circle,
                section,
                offsets,
                weight,
                N=64,
                K=128,
                D=4,
                G=4,
                STATION_ROWS=16,
                PROFILE=1,
                BN=16,
                BK=16,
                BA=8,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return torch.nn.functional.linear(x, weight)

        result["prepared_fused_forward_ms"] = timed(fused, repeats)
        result["prepared_split_forward_ms"] = timed(split, repeats)
        # Capture only the prepared numerical operations, not the routing and
        # validation path. This separates GPU time from Python launch latency.
        result["prepared_fused_gpu_ms"] = do_bench_cudagraph(
            fused, rep=20, return_mode="median"
        )
        result["prepared_split_gpu_ms"] = do_bench_cudagraph(
            split, rep=20, return_mode="median"
        )
        reference = torch.nn.functional.linear(x, layer.dense_weight())
        actual = fused()
        separate = split()
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(separate, reference, atol=3e-5, rtol=3e-5)
        result["max_abs_error"] = (actual - reference).abs().max().item()
    for backend in ("materialized", "tiled", "triton"):
        layer.backend = backend

        def step():
            layer.zero_grad(set_to_none=True)
            x.grad = None
            layer(x).backward(grad)

        result[backend + "_forward_backward_ms"] = timed(step, repeats)
    return result


def main():
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = Path("manifest.json")
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "dtype": "float32",
        "dot_precision": "ieee",
        "repeats": args.repeats,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else None,
        "cases": [],
    }
    for batch, atoms in ((16, 64), (128, 64), (16, 256), (128, 256)):
        case = probe(batch, atoms, args.repeats)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
