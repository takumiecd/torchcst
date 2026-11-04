"""FP32's own long learning trajectory: public eager versus guarded Graph.

No math/kernel/optimizer change. The first eight FP32 updates count toward the
requested total; only capture's temporary updates are restored. Evaluation and
checkpoint boundaries are exercised between epochs. Diagnostics are untimed.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import (
    evaluate,
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.full_step_graph import PreparedStep, ordinary_step, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.full_step_graph_memory import ENDPOINT_PHASE
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans
from torchcst._backends.torch import finite
from torchcst.operators.atom_update import AtomUpdateInputs


def loop_ranges(epoch, steps):
    """All epoch steps exactly once, including the eight real bootstrap updates."""
    if steps < 8:
        raise ValueError("at least eight steps per epoch are required")
    return ((0, 8), (8, steps)) if epoch == 0 else ((0, steps),)


def metrics(totals):
    return [float(totals[0]), int(totals[1])]


def gradient_snapshot(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}


def persist(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def renew_certificates(model, optimizer, inputs, kind):
    """Revalidate live execution after eval, without a numerical update.

    Evaluation can replace the shared fixed-metadata certificate. A training
    forward renews Linear's certificate, but does not renew Polar's certificate.
    Ask the ordinary Dispatcher preparation entrance to validate that binding,
    then build its optional prepared program, just as the optimizer does.
    """
    optimizer.zero_grad(set_to_none=True)
    model(inputs).sum().backward()
    if kind == "cst":
        site = model[2]
        point = site.atoms.p
        group = next(
            g for g in optimizer.param_groups if any(p is point for p in g["params"])
        )
        binding = optimizer._update_bindings[site]
        dispatcher = optimizer._update_dispatcher
        with finite.checking(False):
            dispatcher.prepare(
                binding,
                AtomUpdateInputs(
                    point.detach().clone(), group["lr"], group["amplitude_weight_decay"]
                ),
            )
            program = dispatcher.prepare_optimizer_update(
                binding, optimizer.state_adapter, site
            )
        assert program is not None, "Polar revalidation refused the live cohort"
        binding.__dict__["_optimizer_program"] = program


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
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
    model = make_model(
        init,
        args.kind,
        radius_execution="cached" if args.kind == "cst" else "global-response4",
        execution_plan=(
            plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32
            if args.sort32
            else plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY
            if args.geometry_shared
            else plans.GLOBAL_OWNER_WINDOW_FP32
        )
        if args.kind == "cst"
        else None,
    ).cuda()
    assert torch.get_num_threads() == 1
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer = make_optimizer(model, init, args.kind)
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - started
    initial_test = evaluate(model, vx, vy, 4)
    model.train()
    initial = snapshot(model, optimizer)
    initial_hashes = hashes(initial)
    del initial
    program = None
    total_updates, replay_updates = 0, 0
    captures, epochs, retirements = [], [], []

    def prepare_graph(order, next_step, totals, reason):
        nonlocal preparation_seconds
        torch.cuda.synchronize()
        begin = time.perf_counter()
        if reason != "bootstrap":
            # Renew public execution certificates after an evaluation boundary,
            # without adding an optimizer update or losing counted metrics.
            renew_certificates(
                model,
                optimizer,
                tx[order[4 * next_step : 4 * next_step + 4]],
                args.kind,
            )
        candidate = PreparedStep(model, optimizer, tx, ty, args.kind)
        for dest, source in zip(candidate.totals, totals):
            dest.copy_(source)
        candidate.indices.copy_(order[4 * next_step : 4 * next_step + 4])
        before = snapshot(model, optimizer)
        wanted_totals = metrics(totals)
        candidate.capture(preserve_totals=True)
        after = snapshot(model, optimizer)
        assert hashes(before) == hashes(after), "capture changed live P/Adam state"
        assert metrics(candidate.totals) == wanted_totals, (
            "capture changed counted metrics"
        )
        assert candidate.eligible()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        preparation_seconds += seconds
        captures.append(
            {
                "at_update": total_updates,
                "reason": reason,
                "seconds": seconds,
                "state_fields": len(after),
                "restoration_byte_equal": True,
                "counted_metrics_preserved": True,
            }
        )
        return candidate

    for epoch in range(args.epochs):
        model.train()
        order_cpu = torch.randperm(
            len(tx), generator=torch.Generator().manual_seed(init["seed"] + 100 + epoch)
        )
        order = order_cpu.cuda()
        steps = args.steps_per_epoch or len(tx) // 4
        assert steps <= len(tx) // 4
        totals = (
            (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
            if program is None
            else program.totals
        )
        for t in totals:
            t.zero_()
        # No guard refresh or signature bypass: reprepare only if normal public
        # evaluation changed eligibility, and account for that complete cost.
        if program is not None and not program.eligible():
            retire_begin = time.perf_counter()
            program = None
            gc.collect()
            torch.cuda.synchronize()
            retire_seconds = time.perf_counter() - retire_begin
            preparation_seconds += retire_seconds
            retirements.append(
                {
                    "at_update": total_updates,
                    "seconds": retire_seconds,
                    "reason": "evaluation_boundary",
                }
            )
            program = prepare_graph(order, 0, totals, "evaluation_boundary")
            totals = program.totals
        torch.cuda.synchronize()
        training_seconds = 0.0
        for lo, hi in loop_ranges(epoch, steps):
            if lo == 8 and args.mode == "graph":
                program = prepare_graph(order, lo, totals, "bootstrap")
                totals = program.totals
            torch.cuda.synchronize()
            begin = time.perf_counter()
            enabled = gc.isenabled()
            gc.disable()
            try:
                for step in range(lo, hi):
                    indices = order[4 * step : 4 * step + 4]
                    if program is None:
                        ordinary_step(model, optimizer, tx, ty, indices, totals)
                    else:
                        program.step(indices)
                        assert not program.invalidated, "unexpected eager fallback"
                        replay_updates += 1
                    total_updates += 1
                torch.cuda.synchronize()
            finally:
                if enabled:
                    gc.enable()
            training_seconds += time.perf_counter() - begin
        end = snapshot(model, optimizer)
        grads = gradient_snapshot(model)
        assert all(torch.isfinite(v).all() for v in [*end.values(), *grads.values()])
        values = metrics(totals)
        samples = steps * 4
        train_metrics = {"loss": values[0] / samples, "accuracy": values[1] / samples}
        evaluation_begin = time.perf_counter()
        validation = evaluate(model, vx, vy, 4)
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - evaluation_begin
        model.train()
        assert hashes(snapshot(model, optimizer)) == hashes(end), (
            "evaluation changed numerical training state"
        )
        assert hashes(gradient_snapshot(model)) == hashes(grads), (
            "evaluation changed gradients"
        )
        torch.cuda.synchronize()
        peak_allocated, peak_reserved = (
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        )
        current_allocated, current_reserved = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )
        row = {
            "epoch": epoch + 1,
            "steps": steps,
            "total_updates": total_updates,
            "training_seconds": training_seconds,
            "mean_step_ms": training_seconds * 1000 / steps,
            "train": train_metrics,
            "test": validation,
            "totals": values,
            "state": hashes(end),
            "gradients": hashes(grads),
            "data_order_sha256": hashlib.sha256(
                order_cpu.numpy().tobytes()
            ).hexdigest(),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "end_allocated_bytes": current_allocated,
            "end_reserved_bytes": current_reserved,
            "evaluation_seconds": evaluation_seconds,
            "replay_eligible_after_evaluation": program.eligible() if program else None,
        }
        assert all(
            torch.isfinite(torch.tensor(v))
            for v in [*train_metrics.values(), *validation.values()]
        )
        save_begin = time.perf_counter()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gradients": {name: p.grad for name, p in model.named_parameters()},
                "epoch": epoch + 1,
                "updates": total_updates,
                "totals": values,
                "kind": args.kind,
                "precision": "fp32_body" if args.kind == "cst" else "dense_fp32",
            },
            args.output / f"epoch-{epoch + 1}.pt",
        )
        row["checkpoint_seconds"] = time.perf_counter() - save_begin
        epochs.append(row)
        persist(args.output / "epochs.json", epochs)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "epoch",
                        "total_updates",
                        "mean_step_ms",
                        "train",
                        "test",
                        "replay_eligible_after_evaluation",
                    ]
                }
            ),
            flush=True,
        )
        del end, grads
    torch.cuda.synchronize()
    retire_begin = time.perf_counter()
    program = None
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    retire_seconds = time.perf_counter() - retire_begin
    preparation_seconds += retire_seconds
    retirements.append(
        {
            "at_update": total_updates,
            "seconds": retire_seconds,
            "reason": "final_retirement",
        }
    )
    report = {
        "kind": args.kind,
        "mode": args.mode,
        "seed": init["seed"],
        "batch_size": 4,
        "total_updates": total_updates,
        "replay_updates": replay_updates,
        "unexpected_fallbacks": 0,
        "initialization_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        "initial_state": initial_hashes,
        "initial_test": initial_test,
        "data": info,
        "epochs": epochs,
        "captures": captures,
        "retirements": retirements,
        "geometry_shared": args.geometry_shared or args.sort32,
        "sort32": args.sort32,
        "preparation_seconds": preparation_seconds,
        "training_seconds": sum(e["training_seconds"] for e in epochs),
        "adoption_seconds": preparation_seconds
        + sum(e["training_seconds"] for e in epochs),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resident_allocated_bytes": resident,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cpu_math_threads": torch.get_num_threads(),
        "TF32": False,
        "precision": "FP32 body; FP64 decode/classify/local subtraction/Polar pullback"
        if args.kind == "cst"
        else "FP32 dense",
        "timing_scope": f"epoch training segments include all {total_updates} real updates, cold optimizer, gather/loss/backward/update/metrics and Graph guard/index/version notifications; preparation/capture/restoration/Graph retirement and GC added to adoption_seconds; evaluation/checkpoint excluded",
        "memory_endpoint_phase": ENDPOINT_PHASE,
        "memory_scope": "before optimizer creation; includes initial evaluation, all actual FP32 updates, capture/restoration/repreparation and epoch evaluations; no peak reset after capture; untimed final VJP diagnostics run separately",
        "optimizer_contract": init["optimizer_contract"],
    }
    persist(args.output / "learning.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("cst", "dense"), default="cst")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="smoke only; full experiment uses all15000",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--geometry-shared", action="store_true")
    parser.add_argument("--sort32", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
