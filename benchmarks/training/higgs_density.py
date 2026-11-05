"""HIGGS middle-layer atom-density study; benchmark tooling, not a public API.

All three layers learn. The input/output layers remain dense, and the middle
layer is bias-free in both arms. Dense uses its normal independent initializer;
CST uses the public radius initializer. This is a new protocol, not a replay of
the historical spherical-kernel/B128 experiment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks.training.cell_mass_mnist import (
    make_radius_cst,
    radius_optimizer_contract,
    radius_settings,
    radius_statistics,
    tensor_sha256,
)
from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import exact, ordinary_step, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.full_step_graph_learning import renew_certificates
from benchmarks.training.graph_higgs_cohort import (
    HIGGS_ATOM_DIVISORS,
    HIGGS_GRAPH_COHORT,
    HIGGS_GRAPH_WIDTHS,
)
from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)


def persist(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def atom_count(width, divisor):
    if type(width) is not int or width not in HIGGS_GRAPH_WIDTHS:
        raise ValueError("choose an explicit HIGGS width")
    if type(divisor) is not int or divisor not in HIGGS_ATOM_DIVISORS:
        raise ValueError("choose an explicit HIGGS atom divisor")
    return width * width // divisor


def save_initialization(model, configuration, path):
    """Preserve tensor and extra-state metadata before moving the model to CUDA."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("initialization must be frozen on CPU before CUDA setup")
    torch.save(
        {"model": copy.deepcopy(model.state_dict()), "args": configuration}, path
    )


def make_model(width, divisor, seed, kind):
    count = atom_count(width, divisor)
    if kind not in ("cst", "dense"):
        raise ValueError("kind must be cst or dense")
    # Separate streams keep both ordinary layers identical across all densities
    # and arms, and make the standard dense baseline independent of alpha.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        input_layer = nn.Linear(28, width)
        torch.manual_seed(seed + 2)
        output_layer = nn.Linear(width, 2)
        torch.manual_seed(seed + 3)
        middle = (
            make_radius_cst(
                width,
                count,
                execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
            )
            if kind == "cst"
            else nn.Linear(width, width, bias=False)
        )
    model = nn.Sequential(input_layer, nn.ReLU(), middle, nn.ReLU(), output_layer)
    model._cst_research_cohort = HIGGS_GRAPH_COHORT
    assert all(parameter.requires_grad for parameter in model.parameters())
    return model


def make_optimizer(model, kind, lr):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    contract = radius_optimizer_contract()
    contract["lr"] = lr
    groups = [
        {
            "params": [*model[0].parameters(), *model[4].parameters()],
            "weight_decay": contract["ordinary_weight_decay"],
            "role": "ordinary",
        },
        {
            "params": list(model[2].parameters()),
            "weight_decay": contract["atom_raw_weight_decay"]
            if kind == "cst"
            else contract["dense_weight_decay"],
            "role": "middle",
        },
    ]
    options = {
        name: contract[name]
        for name in ("lr", "fused", "capturable", "betas", "eps", "amsgrad", "maximize")
    }
    if kind == "dense":
        return torch.optim.AdamW(groups, **options), contract
    groups[1]["amplitude_weight_decay"] = contract["amplitude_weight_decay"]
    return CSTAdamW(
        groups,
        model=model,
        check_finite=contract["check_finite"],
        update_selector=FixedSelector(
            RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
        ),
        **options,
    ), contract


def binary_auc(scores, targets):
    """Exact rank AUC with half credit for ties; independent of sklearn."""
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("AUC requires one score and label per example")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("AUC requires finite scores and binary labels")
    positive = int(targets.sum())
    negative = len(targets) - positive
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    average_ranks = np.repeat((starts + 1 + stops) / 2, stops - starts)
    rank_sum = average_ranks[targets[order] == 1].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


@torch.no_grad()
def evaluate(model, x, y):
    """Native B4 evaluation; no dense CST W export or inference shortcut."""
    if len(y) % 4:
        raise ValueError("evaluation rows must be a multiple of four")
    was_training = model.training
    model.eval()
    logits = torch.cat([model(x[i : i + 4]) for i in range(0, len(y), 4)])
    loss = float(nn.functional.cross_entropy(logits, y))
    accuracy = float((logits.argmax(1) == y).float().mean())
    result = {
        "loss": loss,
        "accuracy": accuracy,
        "roc_auc": binary_auc(
            (logits[:, 1] - logits[:, 0]).cpu().numpy(), y.cpu().numpy()
        ),
        "examples": len(y),
        "logit_rms": float(logits.square().mean().sqrt()),
    }
    model.train(was_training)
    return result


@torch.no_grad()
def initial_activations(model, x):
    value = x[:4]
    result = {}
    for index, module in enumerate(model):
        value = module(value)
        result[str(index)] = {
            "rms": float(value.square().mean().sqrt()),
            "max_abs": float(value.abs().max()),
            "shape": list(value.shape),
        }
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.epochs < 1 or args.validation_rows < 4 or args.validation_rows % 4:
        raise ValueError("positive epochs and a B4-aligned validation size required")
    data_bytes = hashlib.sha256(args.data.read_bytes()).hexdigest()
    with np.load(args.data, allow_pickle=False) as data:
        x_train = torch.from_numpy(data["x_train"].copy())
        y_train = torch.from_numpy(data["y_train"].copy())
        x_val = torch.from_numpy(data["x_val"][: args.validation_rows].copy())
        y_val = torch.from_numpy(data["y_val"][: args.validation_rows].copy())
    if len(x_val) != args.validation_rows:
        raise ValueError("requested validation rows unavailable")
    if x_train.dtype != torch.float32 or y_train.dtype != torch.int64:
        raise ValueError("prepared data dtype mismatch")
    model = make_model(args.width, args.divisor, args.seed, args.kind)
    save_initialization(
        model,
        vars(args) | {"data": str(args.data), "output": str(args.output)},
        args.output / "initialization.pt",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx, ty, vx, vy = x_train.cuda(), y_train.cuda(), x_val.cuda(), y_val.cuda()
        model = model.cuda()
        activation = initial_activations(model, tx)
        initial_validation = evaluate(model, vx, vy)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        optimizer, contract = make_optimizer(model, args.kind, args.lr)
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - begin
        epochs, captures, program = [], [], None
        updates, replays = 0, 0
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
        training_seconds = 0.0
        for epoch in range(args.epochs):
            order_cpu = torch.randperm(
                len(tx),
                generator=torch.Generator().manual_seed(args.seed + 100 + epoch),
            )
            order = order_cpu.cuda()
            steps = args.steps_per_epoch or len(tx) // 4
            if steps < 8 or steps > len(tx) // 4:
                raise ValueError("steps_per_epoch must be in [8, training_rows/4]")
            if program is not None:
                # Every evaluation is a declared boundary. Rebuild rather than
                # assuming that its execution certificate remains current.
                torch.cuda.synchronize()
                begin = time.perf_counter()
                program = None
                gc.collect()
                renew_certificates(model, optimizer, tx[order[:4]], args.kind)
                torch.cuda.synchronize()
                preparation_seconds += time.perf_counter() - begin
            for value in totals:
                value.zero_()
            epoch_seconds = 0.0
            ranges = [(0, 8), (8, steps)] if epoch == 0 else [(0, steps)]
            for lo, hi in ranges:
                if lo == hi:
                    continue
                if args.mode == "graph" and (lo == 8 or epoch > 0):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    candidate = FlatPreparedStep(model, optimizer, tx, ty, args.kind)
                    assert not candidate.invalidated, candidate.qualification_error
                    candidate.indices.copy_(order[4 * lo : 4 * lo + 4])
                    for dest, source in zip(candidate.totals, totals):
                        dest.copy_(source)
                    before = snapshot(model, optimizer)
                    assert candidate.capture(preserve_totals=True), "capture refused"
                    restored = exact(before, snapshot(model, optimizer))
                    assert restored["all_byte_equal"], restored
                    assert candidate.eligible(), "captured cohort not eligible"
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    preparation_seconds += seconds
                    program, totals = candidate, candidate.totals
                    captures.append(
                        {
                            "at_update": updates,
                            "seconds": seconds,
                            "restoration": restored,
                            "entry": getattr(candidate, "fp32_entry", None),
                            "qualification_backend": str(
                                candidate._qualification.native_backend
                            ),
                        }
                    )
                    del before, candidate
                torch.cuda.synchronize()
                begin = time.perf_counter()
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    for step in range(lo, hi):
                        indices = order[4 * step : 4 * step + 4]
                        if program is None:
                            ordinary_step(model, optimizer, tx, ty, indices, totals)
                        else:
                            program.step(indices)
                            assert not program.invalidated, "unexpected eager fallback"
                            replays += 1
                        updates += 1
                    torch.cuda.synchronize()
                finally:
                    if gc_enabled:
                        gc.enable()
                epoch_seconds += time.perf_counter() - begin
            training_seconds += epoch_seconds
            state = snapshot(model, optimizer)
            gradients = {
                name: p.grad.detach().cpu().clone()
                for name, p in model.named_parameters()
            }
            assert all(
                bool(torch.isfinite(value).all())
                for value in [*state.values(), *gradients.values()]
            )
            assert all(
                int(value["step"].item()) == updates
                for value in optimizer.state.values()
            )
            peak = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            begin = time.perf_counter()
            validation = evaluate(model, vx, vy)
            torch.cuda.synchronize()
            evaluation_seconds = time.perf_counter() - begin
            assert exact(state, snapshot(model, optimizer))["all_byte_equal"]
            assert all(
                torch.equal(gradients[name], p.grad.detach().cpu())
                for name, p in model.named_parameters()
            )
            row = {
                "epoch": epoch + 1,
                "updates": updates,
                "replays": replays,
                "training_seconds": epoch_seconds,
                "mean_step_ms": epoch_seconds * 1000 / steps,
                "train_loss": float(totals[0]) / (steps * 4),
                "train_accuracy": int(totals[1]) / (steps * 4),
                "validation": validation,
                "evaluation_seconds": evaluation_seconds,
                "training_peak_bytes": peak,
                "state": hashes(state),
                "gradients": hashes(gradients),
                "order_sha256": tensor_sha256(order_cpu),
            }
            if args.kind == "cst":
                row["radius"] = radius_statistics(model[2])
            if args.checkpoint_policy == "full" or epoch == args.epochs - 1:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "gradients": gradients,
                        "updates": updates,
                    },
                    args.output / f"epoch-{epoch + 1}.pt",
                )
            epochs.append(row)
            persist(args.output / "epochs.json", epochs)
            persist(args.output / "captures.json", captures)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "epoch",
                            "updates",
                            "mean_step_ms",
                            "train_loss",
                            "train_accuracy",
                            "validation",
                        )
                    }
                ),
                flush=True,
            )
            del state, gradients
        torch.cuda.synchronize()
        begin = time.perf_counter()
        program = None
        gc.collect()
        torch.cuda.synchronize()
        preparation_seconds += time.perf_counter() - begin
        extension = None
        if args.kind == "cst":
            path = Path(executor._extension().__file__)
            extension = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        result = {
            "schema_version": 1,
            "cohort": HIGGS_GRAPH_COHORT,
            "kind": args.kind,
            "width": args.width,
            "divisor": args.divisor,
            "alpha": 1 / args.divisor,
            "atom_count": atom_count(args.width, args.divisor)
            if args.kind == "cst"
            else None,
            "seed": args.seed,
            "lr": args.lr,
            "mode": args.mode,
            "architecture": f"28-{args.width}-{args.width}-2; ReLU; bias-free middle in both arms",
            "all_layers_trainable": True,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "training_rows": len(tx),
            "validation_rows": len(vy),
            "batch_size": 4,
            "dataset_sha256": data_bytes,
            "initialization_sha256": hashlib.sha256(
                (args.output / "initialization.pt").read_bytes()
            ).hexdigest(),
            "initialization_policy": "public stock radius CST; standard independent nn.Linear dense; common ordinary layers; no gain calibration",
            "initial_activations": activation,
            "initial_validation": initial_validation,
            "kernel_settings": radius_settings() if args.kind == "cst" else None,
            "optimizer_contract": contract,
            "epochs": epochs,
            "captures": captures,
            "total_updates": updates,
            "graph_replays": replays,
            "training_seconds": training_seconds,
            "preparation_seconds": preparation_seconds,
            "adoption_seconds": training_seconds + preparation_seconds,
            "training_peak_allocated_bytes": max(
                e["training_peak_bytes"]["allocated"] for e in epochs
            ),
            "training_peak_reserved_bytes": max(
                e["training_peak_bytes"]["reserved"] for e in epochs
            ),
            "memory_scope": "allocator peaks reset before optimizer creation; includes bootstrap/capture/restoration, persistent input+validation data, and prior epoch evaluations; no dense W/dW in CST; not total process memory",
            "timing_scope": "synchronized real gather/forward/loss/backward/update/metrics; adoption adds optimizer preparation/capture/restoration/retirement; evaluation/checkpoint/hash diagnostics excluded",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cpu_math_threads": 1,
            "TF32": False,
            "extension": extension,
            "limitations": "one seed pilot is not alpha-min confirmation; native B4 only; different kernel/batch/bias from historical HIGGS; test not evaluated",
        }
        persist(args.output / "learning.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--width", type=int, choices=HIGGS_GRAPH_WIDTHS, default=1024)
    parser.add_argument("--divisor", type=int, choices=HIGGS_ATOM_DIVISORS, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--validation-rows", type=int, default=8192)
    parser.add_argument(
        "--checkpoint-policy", choices=("full", "final"), default="final"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
