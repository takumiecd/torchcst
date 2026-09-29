"""Paired production-path measurements with split reductions enabled/disabled."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch

from benchmarks.cuda.linear.benchmark_triton_linear import model, timed
from torchcst.nn._backends._triton import forward as triton_forward


def execution_mode(name):
    return patch(
        "torchcst.nn._backends._triton.forward",
        partial(triton_forward, split_reductions=name == "optimized"),
    )


def paired_wall(fn, repeats):
    """Alternate order to expose shared-host latency and measurement drift."""
    samples = {"baseline": [], "optimized": []}
    for repeat in range(repeats + 3):
        order = ("baseline", "optimized") if repeat % 2 else ("optimized", "baseline")
        for name in order:
            context = execution_mode(name)
            with context:
                torch.cuda.synchronize()
                start = perf_counter()
                fn()
                torch.cuda.synchronize()
                elapsed = (perf_counter() - start) * 1000
                if repeat >= 3:
                    samples[name].append(elapsed)
    return {
        name: {"median_ms": median(values), "samples_ms": values}
        for name, values in samples.items()
    }


def probe(batch, atoms, rows, columns, repeats):
    from triton.testing import do_bench_cudagraph

    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    x = torch.randn(batch, columns, device="cuda", requires_grad=True)
    grad = torch.randn(batch, rows, device="cuda")

    @torch.no_grad()
    def forward():
        return layer(x)

    def training():
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), grad)
        return y, dx, dp

    result = {"batch": batch, "atoms": atoms, "shape": [rows, columns]}
    reference = None
    for name in ("baseline", "optimized"):
        context = execution_mode(name)
        with context:
            actual = training()
            if reference is None:
                reference = tuple(t.detach().clone() for t in actual)
            else:
                for got, expected in zip(actual, reference):
                    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
                result["max_errors_y_dx_dp"] = [
                    (got - expected).abs().max().item()
                    for got, expected in zip(actual, reference)
                ]
            measurements = {}
            for operation, fn in (("forward", forward), ("forward_backward", training)):
                measurements[operation + "_wall_ms"] = timed(fn, repeats)
                # The public contract supports forward capture. Full training
                # capture is not supported by all preparation/autograd paths.
                if operation == "forward":
                    measurements[operation + "_graph_ms"] = do_bench_cudagraph(
                        fn, rep=30, return_mode="median"
                    )
            result[name] = measurements
    result["interleaved_wall"] = {
        "forward": paired_wall(forward, repeats),
        "forward_backward": paired_wall(training, repeats),
    }
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
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "precision": "float32 IEEE",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else None,
        "repeats": args.repeats,
        "cases": [],
    }
    for args_case in (
        (16, 64, 64, 128),
        (128, 64, 64, 128),
        (16, 256, 64, 128),
        (128, 256, 64, 128),
        (32, 512, 256, 512),
        (128, 512, 256, 512),
    ):
        case = probe(*args_case, args.repeats)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
