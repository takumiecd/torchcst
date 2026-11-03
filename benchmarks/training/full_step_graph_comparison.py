"""One fresh-process cell of the bounded full-step Graph comparison."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.full_step_graph import (
    PreparedStep,
    ordinary_step,
    restore,
    snapshot,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans


def hashes(values):
    return {
        k: {
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "sha256": hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest(),
        }
        for k, v in values.items()
    }


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    init = torch.load(
        args.inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    data, info = load_mnist(args.inputs / "data")
    tx = normalize_images(
        data["train_x"],
        "cuda",
        info["mean"],
        info["std"],
        data_normalization="original",
    )
    vx = normalize_images(
        data["t10k_x"], "cuda", info["mean"], info["std"], data_normalization="original"
    )
    ty, vy = data["train_y"].cuda(), data["t10k_y"].cuda()
    order_cpu = torch.randperm(
        len(tx), generator=torch.Generator().manual_seed(init["seed"] + 100)
    )
    order = order_cpu.cuda()
    model = make_model(
        init,
        args.kind,
        radius_execution="cached" if args.kind == "cst" else "global-response4",
        execution_plan=plans.GLOBAL_OWNER_WINDOW_RECOMPUTE
        if args.kind == "cst"
        else None,
    ).cuda()
    checkpoint = None
    if args.state == "final":
        checkpoint = torch.load(
            args.inputs
            / ("baseline_final.pt" if args.kind == "cst" else "dense_final.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    setup_start = time.perf_counter()
    optimizer = make_optimizer(model, init, args.kind)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        wanted = snapshot(model, optimizer)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    for i in range(8):
        ordinary_step(model, optimizer, tx, ty, order[4 * i : 4 * i + 4], totals)
    if checkpoint is not None:
        restore(model, optimizer, wanted)
    start = snapshot(model, optimizer)
    # All precision conditions start after the SAME eight FP64 updates. Prepare
    # the candidate's guarded public entrance without changing P or Adam state.
    if args.kind == "cst" and args.precision == "fp32":
        from torchcst import FixedSelector, make_algorithm_registry

        model[2].selector = FixedSelector(
            plans.GLOBAL_OWNER_WINDOW_FP32, registry=make_algorithm_registry()
        )
        optimizer.zero_grad(set_to_none=True)
        model(tx[order[:4]]).sum().backward()
        # PreparedStep checks the current gradient metadata before its first
        # body clears gradients. Keep these preparation gradients alive, just
        # as the FP64 common warmup does; P and Adam state remain unchanged.
    program = None
    if args.mode != "eager":
        program = PreparedStep(model, optimizer, tx, ty, args.kind)
        if args.mode == "graph":
            program.capture()
        totals = program.totals
    for total in totals:
        total.zero_()
    gc.collect()
    torch.cuda.synchronize()
    actual_start = snapshot(model, optimizer)
    assert hashes(actual_start) == hashes(start), (
        "capture/restore changed live start state"
    )
    start = actual_start
    setup_seconds = time.perf_counter() - setup_start
    started = time.perf_counter()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(args.steps):
            indices = order[4 * i : 4 * i + 4]
            if program is None:
                ordinary_step(model, optimizer, tx, ty, indices, totals)
            else:
                program.step(indices)
        torch.cuda.synchronize()
    finally:
        if gc_enabled:
            gc.enable()
    seconds = time.perf_counter() - started
    peak_allocated, peak_reserved = (
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )
    assert program is None or not program.invalidated, "unexpected eager fallback"
    end = snapshot(model, optimizer)
    gradients = {
        name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
    }
    row = {
        "kind": args.kind,
        "precision": args.precision if args.kind == "cst" else "dense_fp32",
        "mode": args.mode,
        "state": args.state,
        "repeat": args.repeat,
        "steps": args.steps,
        "seconds": seconds,
        "mean_step_ms": seconds * 1000 / args.steps,
        "examples_per_second": 4 * args.steps / seconds,
        "setup_seconds": setup_seconds,
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "extra_peak_allocated_bytes": peak_allocated - resident,
        "memory_scope": "before optimizer construction/loading, warmup, workspace/capture/restoration through replay; no peak reset after capture",
        "timing_scope": "256-step block, gather/loss/all backward/optimizer/metrics plus replay guard/index copy/version notification, boundary synchronization only",
        "start": hashes(start),
        "end": hashes(end),
        "gradients": hashes(gradients),
        "totals": [float(totals[0]), int(totals[1])],
        "finite": all(
            bool(torch.isfinite(v).all()) for v in [*end.values(), *gradients.values()]
        ),
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "initialization_sha256": hashlib.sha256(
            (args.inputs / "initialization.pt").read_bytes()
        ).hexdigest(),
        "data": info,
        "resident_validation_shapes": [list(vx.shape), list(vy.shape)],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "stream": torch.cuda.current_stream().cuda_stream,
    }
    assert row["finite"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    if args.repeat == 0:
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            args.output.with_suffix(".pt"),
        )
    print(
        {
            k: row[k]
            for k in (
                "kind",
                "mode",
                "state",
                "mean_step_ms",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--mode", choices=("eager", "prepared", "graph"), required=True)
    parser.add_argument("--state", choices=("initial", "final"), required=True)
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
