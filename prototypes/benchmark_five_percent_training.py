"""Dense versus trainable native CST at 5% atom count.

This uses the production Strip CSTLinear, whose geometry differs from the
mapped BlockStripLinear forward prototype. It measures a complete AdamW step,
including fresh CST preparation, input/parameter gradients and optimizer work.
Run each mode in a separate process so peak memory has an interpretable scope.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check, dense_control
from prototypes.benchmark_triton_linear import model
from torchcst.nn._backends._preparation import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 4096), required=True)
    parser.add_argument("--mode", choices=("dense", "cst"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--microbatch-rows", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    if min(args.repeats, args.microbatch_rows, args.accumulation_steps) < 1:
        parser.error("repeats, microbatch-rows and accumulation-steps must be positive")
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    n = args.size
    atom_count = round(n * n * 0.05)
    layer = model(atom_count, rows=n, columns=n)
    print(
        json.dumps({"stage": "created", "mode": args.mode, "atoms": atom_count}),
        flush=True,
    )
    with torch.no_grad():
        prepared = prepare(layer, layer.atoms.p, support_layout=True)
        weight, canonical = dense_control(layer, prepared)
    assert canonical["passed"], canonical
    del prepared
    x = torch.randn(args.microbatch_rows, n, device="cuda", requires_grad=True)
    gradient = torch.randn(args.microbatch_rows, n, device="cuda")
    gradient.div_(args.accumulation_steps)
    with torch.no_grad():
        expected = F.linear(x, weight)
        if args.mode == "cst":
            output_check = check(layer(x), expected)
            assert output_check["passed"], output_check
        else:
            output_check = check(expected, expected)
    if args.mode == "dense":
        parameter = torch.nn.Parameter(weight.detach())
        del layer, weight
        forward = lambda: F.linear(x, parameter)
    else:
        parameter = layer.atoms.p
        del weight
        forward = lambda: layer(x)
    del expected
    gc.collect()
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=True)

    def step(*, record_phases=False):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        phases = []
        for microstep in range(args.accumulation_steps):
            output = forward()
            if record_phases:
                torch.cuda.synchronize()
                phases.append(
                    {
                        "microstep": microstep,
                        "phase": "forward",
                        "live_bytes": torch.cuda.memory_allocated(),
                        "peak_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
            output.backward(gradient)
            if record_phases:
                torch.cuda.synchronize()
                phases.append(
                    {
                        "microstep": microstep,
                        "phase": "backward",
                        "live_bytes": torch.cuda.memory_allocated(),
                        "peak_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
            del output
        optimizer.step()
        if record_phases:
            torch.cuda.synchronize()
            phases.append(
                {
                    "microstep": args.accumulation_steps - 1,
                    "phase": "optimizer",
                    "live_bytes": torch.cuda.memory_allocated(),
                    "peak_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        return phases

    # Include the first optimizer state creation in the report, then measure
    # steady-state steps separately. Each mode starts from the same seed.
    torch.cuda.synchronize()
    first_start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first_step_ms = (time.perf_counter() - first_start) * 1000
    print(
        json.dumps({"stage": "first_step", "mode": args.mode, "ms": first_step_ms}),
        flush=True,
    )
    samples = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    optimizer.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    phases = step(record_phases=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "torch": torch.__version__,
        "mode": args.mode,
        "shape": [args.microbatch_rows, n, n],
        "microbatch_rows": args.microbatch_rows,
        "accumulation_steps": args.accumulation_steps,
        "effective_rows_per_step": args.microbatch_rows * args.accumulation_steps,
        "atoms": atom_count,
        "atom_parameter_count_fraction": atom_count / (n * n),
        "canonical": canonical,
        "output_check": output_check,
        "parameter_bytes": parameter.numel() * parameter.element_size(),
        "input_requires_grad": True,
        "optimizer": "torch.optim.AdamW foreach=True lr=1e-3",
        "dense_weight_creation_timed": False,
        "cst_preparation_timed": args.mode == "cst",
        "first_step_ms": first_step_ms,
        "steady_step_samples_ms": samples,
        "steady_step_median_ms": statistics.median(samples),
        "warmed_baseline_allocated_bytes": baseline,
        "warmed_step_peak_allocated_bytes": peak,
        "warmed_step_peak_increment_bytes": peak - baseline,
        "warmed_step_phases": phases,
        "memory_scope": "Run one mode per process. Baseline includes model, input and initialized AdamW state; step peak includes gradients and temporaries. CUDA allocator reserved bytes are excluded.",
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "completed",
                "mode": args.mode,
                "median_ms": result["steady_step_median_ms"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
