"""Fresh-process initial8 reference for the bounded guard-cost screen.

This compares historical R0 with topology-corrected G0, not a speed candidate.
Both use identical #132 arithmetic. A reference module path selects historical
PreparedStep without patching guards or exposing unchecked replay.
"""

import argparse
import gc
import hashlib
import importlib.util
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
from benchmarks.training.full_step_graph import PreparedStep, ordinary_step, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans


def historical_prepared_step(path):
    spec = importlib.util.spec_from_file_location("guard_reference_r0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PreparedStep


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    loading_begin = time.perf_counter()
    init_path = args.inputs / "initialization.pt"
    init = torch.load(init_path, map_location="cpu", weights_only=True)
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
        execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
        if args.kind == "cst"
        else None,
    ).cuda()
    step_type = (
        PreparedStep
        if args.reference is None
        else historical_prepared_step(args.reference)
    )
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter() - loading_begin
    resident = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    # One reset before optimizer/bootstrap/capture. Never empty_cache or reset
    # peaks inside that measured lifetime.
    torch.cuda.reset_peak_memory_stats()
    setup_begin = time.perf_counter()
    optimizer = make_optimizer(model, init, args.kind)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    for i in range(8):
        ordinary_step(model, optimizer, tx, ty, order[4 * i : 4 * i + 4], totals)
    start = snapshot(model, optimizer)
    torch.cuda.synchronize()
    bootstrap_end = time.perf_counter()
    program = step_type(model, optimizer, tx, ty, args.kind)
    torch.cuda.synchronize()
    preparation_end = time.perf_counter()
    program.capture()
    torch.cuda.synchronize()
    capture_end = time.perf_counter()
    totals = program.totals
    for value in totals:
        value.zero_()
    gc.collect()
    torch.cuda.synchronize()
    actual_start = snapshot(model, optimizer)
    assert hashes(actual_start) == hashes(start), "capture changed initial8 state"
    start = actual_start
    assert all(int(state["step"].item()) == 8 for state in optimizer.state.values())
    setup_end = time.perf_counter()
    started = time.perf_counter()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(256):
            program.step(order[4 * i : 4 * i + 4])
        enqueued = time.perf_counter()
        torch.cuda.synchronize()
        completed = time.perf_counter()
    finally:
        if gc_enabled:
            gc.enable()
    assert not program.invalidated, "unexpected eager fallback"
    assert all(int(state["step"].item()) == 264 for state in optimizer.state.values())
    peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    current = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    end = snapshot(model, optimizer)
    gradients = {
        name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
    }
    metrics = [float(totals[0]), int(totals[1])]
    # Save actual tensors for direct retrieval verification, outside timing.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"start": start, "end": end, "gradients": gradients, "metrics": metrics},
        args.output.with_suffix(".pt"),
    )
    extension = None
    if args.kind == "cst":
        path = Path(program.extension.__file__)
        extension = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    retirement_begin = time.perf_counter()
    program = None
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    retirement_seconds = time.perf_counter() - retirement_begin
    retired = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    row = {
        "reference": "G0" if args.reference is None else "R0",
        "kind": args.kind,
        "repeat": args.repeat,
        "steps": 256,
        "seconds": completed - started,
        "mean_step_ms": (completed - started) * 1000 / 256,
        "enqueue_seconds": enqueued - started,
        "final_sync_seconds": completed - enqueued,
        "loading_seconds": loading_seconds,
        "bootstrap_seconds": bootstrap_end - setup_begin,
        "prepared_constructor_seconds": preparation_end - bootstrap_end,
        "capture_restore_seconds": capture_end - preparation_end,
        "post_capture_validation_seconds": setup_end - capture_end,
        "setup_seconds": setup_end - setup_begin,
        "retirement_seconds": retirement_seconds,
        "resident_bytes": resident,
        "capture_inclusive_peak_bytes": peak,
        "after_updates_current_bytes": current,
        "retired_current_bytes": retired,
        "memory_v2_1_admitted": False,
        "unexpected_fallbacks": 0,
        "clock_start": 8,
        "clock_end": 264,
        "start": hashes(start),
        "end": hashes(end),
        "gradients": hashes(gradients),
        "metrics": metrics,
        "finite": all(
            bool(torch.isfinite(v).all()) for v in [*end.values(), *gradients.values()]
        ),
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "initialization_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        "reference_source_sha256": hashlib.sha256(
            args.reference.read_bytes()
        ).hexdigest()
        if args.reference
        else None,
        "data": info,
        "resident_validation_shapes": [list(vx.shape), list(vy.shape)],
        "extension": extension,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "timing_scope": "uninstrumented 256 complete steps; guard/index gather/loss/backward/optimizer/metrics/version notification and final sync; no per-step synchronization",
        "memory_scope": "one reset before optimizer/bootstrap/capture; no empty_cache; capture through update peaks, retirement currents separately; not lifecycle-v2.1 admission",
    }
    assert row["finite"]
    args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    print(
        {
            key: row[key]
            for key in (
                "reference",
                "kind",
                "repeat",
                "mean_step_ms",
                "enqueue_seconds",
                "final_sync_seconds",
            )
        },
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
