"""CST versus a stored dense weight: the primary performance reference.

Dense weights are constructed once outside timing. CST preparation remains
inside every call. Training includes dX and dW/dP, but no optimizer step.
BF16 dense is a separate practical speed target, not an equal-precision claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F

from benchmarks.cuda.linear.benchmark_triton_linear import model
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan


def interleaved(functions, repeats):
    names = list(functions)
    samples = {name: [] for name in names}
    for iteration in range(repeats + 3):
        # Rotate and reverse so no implementation always gets the same slot.
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        if iteration % 2:
            order = order[::-1]
        for name in order:
            torch.cuda.synchronize()
            start = perf_counter()
            functions[name]()
            torch.cuda.synchronize()
            if iteration >= 3:
                samples[name].append((perf_counter() - start) * 1000)
    return {
        name: {"median_ms": median(values), "samples_ms": values}
        for name, values in samples.items()
    }


def peak_extra_bytes(fn):
    """Active tensor allocations above the warmed baseline; excludes pool reserve."""
    fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    del output
    return extra


def storage_bytes(tensors):
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors
    }
    return sum(storages.values())


def probe(batch, atoms, rows, columns, repeats):
    from triton.testing import do_bench_cudagraph

    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    # Dense has no CST reconstruction or routing in its measured path.
    with torch.no_grad():
        weight = layer.dense_weight().detach().contiguous()
    weight.requires_grad_()
    weight_bf16 = weight.detach().to(torch.bfloat16).requires_grad_()
    x = torch.randn(batch, columns, device="cuda", requires_grad=True)
    x_bf16 = x.detach().to(torch.bfloat16).requires_grad_()
    gradient = torch.randn(batch, rows, device="cuda")
    gradient_bf16 = gradient.to(torch.bfloat16)

    def dense_forward():
        with torch.no_grad():
            return F.linear(x, weight)

    def dense_bf16_forward():
        with torch.no_grad():
            return F.linear(x_bf16, weight_bf16)

    def cst_forward():
        with torch.no_grad():
            return layer(x)

    def dense_training():
        return torch.autograd.grad(F.linear(x, weight), (x, weight), gradient)

    def dense_bf16_training():
        return torch.autograd.grad(
            F.linear(x_bf16, weight_bf16), (x_bf16, weight_bf16), gradient_bf16
        )

    def cst_training():
        return torch.autograd.grad(layer(x), (x, layer.atoms.p), gradient)

    forward = {
        "dense_fp32": dense_forward,
        "dense_bf16": dense_bf16_forward,
        "cst_fp32": cst_forward,
    }
    training = {
        "dense_fp32": dense_training,
        "dense_bf16": dense_bf16_training,
        "cst_fp32": cst_training,
    }
    expected, actual = dense_forward(), cst_forward()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    expected_dx, actual_dx = dense_training()[0], cst_training()[0]
    torch.testing.assert_close(actual_dx, expected_dx, atol=3e-5, rtol=3e-5)
    low_precision_error = dense_bf16_forward().float() - expected
    errors = {
        "cst_output_max_abs": (actual - expected).abs().max().item(),
        "cst_dx_max_abs": (actual_dx - expected_dx).abs().max().item(),
        "bf16_output_max_abs": low_precision_error.abs().max().item(),
        "bf16_output_relative_l2": (
            low_precision_error.norm() / expected.norm().clamp_min(1e-30)
        ).item(),
    }
    del expected, actual, expected_dx, actual_dx, low_precision_error
    plan = execution_plan(layer)
    constants = list(layer.buffers()) + [plan.circle, plan.section]
    constants += [getattr(plan.routing, field.name) for field in fields(plan.routing)]
    parameter_bytes = {
        "dense_fp32": weight.numel() * weight.element_size(),
        "dense_bf16": weight_bf16.numel() * weight_bf16.element_size(),
        "cst_fp32": layer.atoms.p.numel() * layer.atoms.p.element_size(),
    }
    result = {
        "batch": batch,
        "atoms": atoms,
        "shape": [rows, columns],
        "chart_tile_shape": list(layer.chart.tile_shape),
        "errors": errors,
        "parameter_bytes": parameter_bytes,
        "cst_fixed_buffer_and_plan_bytes": storage_bytes(constants),
        "forward_wall": interleaved(forward, repeats),
        "forward_backward_wall": interleaved(training, repeats),
        "forward_graph_ms": {},
        "peak_extra_allocated_bytes": {},
    }
    for name, forward_fn in forward.items():
        result["peak_extra_allocated_bytes"][name] = {
            "forward": peak_extra_bytes(forward_fn),
            "forward_backward": peak_extra_bytes(training[name]),
        }
        result["forward_graph_ms"][name] = do_bench_cudagraph(
            forward_fn, rep=30, return_mode="median"
        )
    result["time_ratio_cst_over_dense"] = {}
    for name in ("dense_fp32", "dense_bf16"):
        result["time_ratio_cst_over_dense"][name] = {
            "forward_wall": result["forward_wall"]["cst_fp32"]["median_ms"]
            / result["forward_wall"][name]["median_ms"],
            "forward_backward_wall": result["forward_backward_wall"]["cst_fp32"][
                "median_ms"
            ]
            / result["forward_backward_wall"][name]["median_ms"],
            "forward_graph": result["forward_graph_ms"]["cst_fp32"]
            / result["forward_graph_ms"][name],
        }
    return result


def main():
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_commit": args.source_commit,
        "fp32_tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
        "repeats": args.repeats,
        "bias": False,
        "optimizer_step": False,
        "dense_weight_creation_timed": False,
        "cst_preparation_timed": True,
        "memory_scope": "Parameter bytes separately; warmed peak active allocation delta includes temporaries, outputs and gradients, excludes allocator reserve and optimizer state.",
        "cases": [],
    }
    for case in (
        (16, 64, 64, 128),
        (128, 64, 64, 128),
        (16, 256, 64, 128),
        (128, 256, 64, 128),
        (32, 512, 256, 512),
        (128, 512, 256, 512),
    ):
        result = probe(*case, args.repeats)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
