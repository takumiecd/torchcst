"""Large Strip+Torus forward scaling, with the same FP32 W for every path.

Dense W is materialized once outside timing. Canonical all-atom spot checks
validate that control independently of the I/B generator. Full output checks
then gate every timed CST path. This measures forward, not training quality.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from benchmarks.cuda.linear.benchmark_triton_linear import model
from experiments.cuda.linear.local_atom_linear import LocalAtom
from experiments.cuda.linear.local_atom_linear import forward as direct
from torchcst.nn._backends._preparation import prepare
from torchcst.nn._backends._triton import _FusedLinear
from torchcst.nn._backends._triton import forward as fused
from torchcst.nn._backends._triton_kernels import materialize_weights


def log(**values):
    print(json.dumps(values), flush=True)


def check(actual, expected):
    error = (actual - expected).abs()
    limit = 3e-5 + 3e-5 * expected.abs()
    return {
        "passed": bool(torch.all(error <= limit)),
        "max_abs": error.max().item(),
        "relative_l2": (error.norm() / expected.norm().clamp_min(1e-30)).item(),
        "violations": int((error > limit).sum()),
        "atol": 3e-5,
        "rtol": 3e-5,
    }


def dense_control(layer, prepared):
    packed, circle, section, offsets = prepared
    n, k = layer.out_features, layer.in_features
    weight = torch.empty((n, k), device="cuda")
    materialize_weights[(n // 16, k // 16)](
        packed,
        circle,
        section,
        offsets,
        weight,
        N=n,
        K=k,
        D=packed.shape[1] - 2,
        G=layer.chart.tile_count,
        STATION_ROWS=16,
        PROFILE=1,
        BN=16,
        BK=16,
        BA=8,
        SUPPORT_LAYOUT=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Random sites plus the largest-magnitude entry of regularly sampled rows.
    # All atoms participate in the canonical calculation, without I/B filtering.
    selected_rows = torch.linspace(0, n - 1, 128, device="cuda").long()
    peak_columns = weight[selected_rows].abs().argmax(dim=1)
    selection = torch.cat(
        (
            torch.randint(n * k, (1024,), device="cuda"),
            selected_rows * k + peak_columns,
        )
    )
    encoded, amplitude, precision = layer.kernel.tile_parameters(
        layer.chart, layer.atoms.p
    )
    expected = []
    for chunk in selection.split(64):
        values = layer.kernel.profile.evaluate_with_precision_slice(
            layer.chart, encoded, precision, chunk
        )
        expected.append((values * amplitude).sum(dim=-1))
    validation = check(weight.flatten()[selection], torch.cat(expected))
    validation["sites"] = selection.numel()
    return weight, validation


def peak_extra(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del y
    return peak


def benchmark_case(layer, weight, prepared, batch, rounds, rep):
    n, k = weight.shape
    log(stage="case_start", shape=[n, k], batch=batch)
    packed, circle, section, offsets = prepared
    bounds = torch.stack((section.amin(0), section.amax(0)))
    x = torch.randn((batch, k), device="cuda")
    p = layer.atoms.p
    functions = {
        "dense": lambda: F.linear(x, weight),
        "fused_bm16": lambda: fused(layer, x, p, batch_tile=16),
        "fused_bm64": lambda: fused(layer, x, p, batch_tile=64),
        "direct_bm16": lambda: direct(layer, x, p, batch_tile=16),
        "fused_prepared_bm64": lambda: _FusedLinear.apply(
            x, packed, circle, section, offsets, 16, 1, False, True, 64
        ),
        "direct_prepared_bm16": lambda: LocalAtom.apply(
            x, packed, circle, section, offsets, bounds, 16, 1, 16, 128
        ),
    }
    expected = functions["dense"]()
    errors = {}
    for name, fn in functions.items():
        actual = fn()
        errors[name] = check(actual, expected)
        del actual
        log(stage="validated", shape=[n, k], batch=batch, path=name, **errors[name])
    del expected
    # Do not present timings of a numerically failing path as valid performance.
    valid = [name for name in functions if errors[name]["passed"]]
    peaks = {name: peak_extra(functions[name]) for name in valid}
    samples = {name: [] for name in valid}
    for round_index in range(rounds):
        names = valid if round_index % 2 == 0 else list(reversed(valid))
        for name in names:
            ms = do_bench_cudagraph(functions[name], rep=rep)
            samples[name].append(ms)
            log(
                stage="timed",
                shape=[n, k],
                batch=batch,
                path=name,
                round=round_index,
                ms=ms,
            )
    return {
        "batch": batch,
        "errors": errors,
        "graph_ms": {name: median(values) for name, values in samples.items()},
        "samples_ms": samples,
        "peak_extra_allocated_bytes": peaks,
        "input_bytes": x.numel() * x.element_size(),
        "output_bytes": batch * n * 4,
        "cuda_blocks": {
            "fused_bm16": (batch + 15) // 16 * layer.chart.tile_count,
            "fused_bm64": (batch + 63) // 64 * layer.chart.tile_count,
            "direct_bm16": (batch + 15) // 16 * n,
        },
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--atoms-per-tile", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=float, default=30)
    parser.add_argument("--small-control", action="store_true")
    args = parser.parse_args()
    assert all(size > 0 and size % 16 == 0 for size in args.sizes)
    assert all(batch > 0 for batch in args.batches)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    result = {
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "total_device_bytes": properties.total_memory,
        "torch": torch.__version__,
        "source_commit": args.source_commit,
        "precision": "FP32 IEEE; TF32 off",
        "timing": "CUDA Graph, preparation included except *_prepared; W precomputed for dense",
        "memory": "Incremental allocated peak of warmed uncaptured forward; includes output, excludes existing inputs/model/oracle; not total training memory",
        "atoms_per_tile": args.atoms_per_tile,
        "seed": 21,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    shapes = [(size, size, args.batches) for size in args.sizes]
    if args.small_control:
        shapes.insert(0, (256, 512, [128]))
    save()
    for n, k, batches in shapes:
        torch.manual_seed(21)
        atoms = n // 16 * args.atoms_per_tile
        log(stage="shape_start", shape=[n, k], atoms=atoms)
        started = time.monotonic()
        layer = model(atoms, rows=n, columns=k)
        prepared = prepare(layer, layer.atoms.p, support_layout=True)
        weight, validation = dense_control(layer, prepared)
        counts = torch.diff(prepared[3]).cpu().tolist()
        shape = {
            "shape": [n, k],
            "tile_shape": list(layer.chart.tile_shape),
            "tile_count": layer.chart.tile_count,
            "atoms": atoms,
            "bucket_counts": counts,
            "interior_atoms": sum(counts[:-1:2]),
            "boundary_atoms": sum(counts[1:-1:2]),
            "idle_atoms": counts[-1],
            "weight_nonzero_fraction": torch.count_nonzero(weight).item()
            / weight.numel(),
            "weight_abs_max": weight.abs().max().item(),
            "atom_parameter_bytes": layer.atoms.p.numel()
            * layer.atoms.p.element_size(),
            "dense_weight_bytes": weight.numel() * weight.element_size(),
            "prepared_tensor_bytes": sum(
                t.numel() * t.element_size() for t in prepared
            ),
            "dense_control_canonical_check": validation,
            "batches": [],
        }
        result["cases"].append(shape)
        save()
        log(
            stage="control_validated",
            shape=[n, k],
            **validation,
            nonzero_fraction=shape["weight_nonzero_fraction"],
        )
        if validation["passed"]:
            for batch in batches:
                shape["batches"].append(
                    benchmark_case(
                        layer, weight, prepared, batch, args.rounds, args.rep_ms
                    )
                )
                save()
        shape["elapsed_seconds"] = time.monotonic() - started
        save()
        del weight, prepared, layer
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    save()
    log(stage="completed", shapes=len(result["cases"]))


if __name__ == "__main__":
    main()
