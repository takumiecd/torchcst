"""Long HIGGS training with dev-only selection and deferred confirmation.

The first 50,000 official validation rows select the checkpoint. The remaining
50,000 rows are evaluated only by the explicit confirmation evaluator. No test
examples or labels are used. All three layers train; only the middle is CST.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks.training.cell_mass_mnist import (
    make_radius_cst,
    radius_settings,
    radius_statistics,
    tensor_sha256,
)
from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import exact, ordinary_step, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.full_step_graph_learning import renew_certificates
from benchmarks.training.graph_higgs_cohort import (
    HIGGS_CONFIRMATION_COHORT,
    HIGGS_CONFIRMATION_DIVISORS,
    HIGGS_CONFIRMATION_WIDTHS,
)
from benchmarks.training.higgs_density import (
    binary_auc,
    initial_activations,
    make_optimizer,
    persist,
    save_initialization,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans

DEV_ROWS = 50_000
CONFIRMATION_ROWS = 50_000


def atom_count(width, divisor):
    if type(width) is not int or width not in HIGGS_CONFIRMATION_WIDTHS:
        raise ValueError("confirmation permits only widths 1024/2048")
    if type(divisor) is not int or divisor not in HIGGS_CONFIRMATION_DIVISORS:
        raise ValueError("confirmation density is outside the declared grid")
    return width * width // divisor


def make_model(width, divisor, seed, kind):
    count = atom_count(width, divisor)
    if kind not in ("cst", "dense"):
        raise ValueError("kind must be cst or dense")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = nn.Linear(28, width)
        torch.manual_seed(seed + 2)
        last = nn.Linear(width, 2)
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
    model = nn.Sequential(first, nn.ReLU(), middle, nn.ReLU(), last)
    model._cst_research_cohort = HIGGS_CONFIRMATION_COHORT
    assert all(p.requires_grad for p in model.parameters())
    return model


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_slice(name):
    if name == "development":
        return slice(0, DEV_ROWS)
    if name == "confirmation":
        return slice(DEV_ROWS, DEV_ROWS + CONFIRMATION_ROWS)
    raise ValueError("unknown validation partition")


def load_partition(path, name):
    """Only the declared validation partition can affect its evaluator."""
    selected = validation_slice(name)
    with np.load(path, allow_pickle=False) as data:
        if data["x_val"].shape != (DEV_ROWS + CONFIRMATION_ROWS, 28):
            raise ValueError("expected the complete official 100k validation payload")
        if data["y_val"].shape != (DEV_ROWS + CONFIRMATION_ROWS,):
            raise ValueError("validation label shape mismatch")
        x = torch.from_numpy(data["x_val"][selected].copy())
        y = torch.from_numpy(data["y_val"][selected].copy())
    if x.dtype != torch.float32 or y.dtype != torch.int64:
        raise ValueError("validation dtype mismatch")
    return x, y


def choose_best(previous, metrics, epoch):
    """Predeclared minimum-development-loss rule; earliest epoch wins ties."""
    loss = metrics["loss"]
    if not np.isfinite(loss) or type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint selection requires finite dev loss/positive epoch")
    return previous is None or loss < previous["development"]["loss"]


def development_rows(args):
    rows = args.development_rows
    if type(rows) is not int or rows < 4 or rows > DEV_ROWS or rows % 4:
        raise ValueError("development rows must be B4-aligned within [4,50000]")
    if rows != DEV_ROWS and not (
        8 <= args.steps_per_epoch <= 32 and 1 <= args.epochs <= 2
    ):
        raise ValueError(
            "reduced development rows are reserved for short qualification"
        )
    return rows


@torch.no_grad()
def evaluate(model, x, y):
    if len(y) % 4:
        raise ValueError("native evaluation requires true B4-aligned rows")
    training = model.training
    model.eval()
    logits = torch.cat([model(x[i : i + 4]) for i in range(0, len(y), 4)])
    cpu_logits, cpu_labels = logits.cpu().numpy(), y.cpu().numpy()
    metrics = {
        "loss": float(nn.functional.cross_entropy(logits, y)),
        "accuracy": float((logits.argmax(1) == y).float().mean()),
        "roc_auc": binary_auc(cpu_logits[:, 1] - cpu_logits[:, 0], cpu_labels),
        "examples": len(y),
        "logit_rms": float(logits.square().mean().sqrt()),
    }
    model.train(training)
    return metrics, cpu_logits, cpu_labels


def _configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _checkpoint(
    model, optimizer, gradients, epoch, updates, development, configuration
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "gradients": gradients,
        "epoch": epoch,
        "updates": updates,
        "development": development,
        "configuration": configuration,
    }


def validate_checkpoint(configuration, args, data_sha):
    expected = {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        "kind": args.kind,
        "width": args.width,
        "divisor": args.divisor,
        "seed": args.seed,
        "dataset_sha256": data_sha,
    }
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint identity differs from the declared evaluation")


def run(args):
    _configure()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    dev_rows = development_rows(args)
    args.output.mkdir(parents=True, exist_ok=False)
    data_sha = file_sha256(args.data)
    with np.load(args.data, allow_pickle=False) as data:
        tx_cpu = torch.from_numpy(data["x_train"].copy())
        ty_cpu = torch.from_numpy(data["y_train"].copy())
    if tuple(tx_cpu.shape) != (100_000, 28) or tuple(ty_cpu.shape) != (100_000,):
        raise ValueError("confirmation protocol requires the 100k training payload")
    if tx_cpu.dtype != torch.float32 or ty_cpu.dtype != torch.int64:
        raise ValueError("training dtype mismatch")
    vx_cpu, vy_cpu = load_partition(args.data, "development")
    vx_cpu, vy_cpu = vx_cpu[:dev_rows], vy_cpu[:dev_rows]
    model = make_model(args.width, args.divisor, args.seed, args.kind)
    configuration = {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        "kind": args.kind,
        "width": args.width,
        "divisor": args.divisor,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "dataset_sha256": data_sha,
    }
    save_initialization(
        model,
        vars(args)
        | {
            "data": str(args.data),
            "output": str(args.output),
            "evaluate_only_checkpoint": None,
            "evaluation_output": None,
        },
        args.output / "initialization.pt",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx, ty = tx_cpu.cuda(), ty_cpu.cuda()
        vx, vy = vx_cpu.cuda(), vy_cpu.cuda()
        model = model.cuda()
        initial_activation = initial_activations(model, tx)
        initial_development, _, _ = evaluate(model, vx, vy)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        optimizer, contract = make_optimizer(model, args.kind, args.lr)
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - begin
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
        epochs, captures, best = [], [], None
        updates, replays, training_seconds, program = 0, 0, 0.0, None
        steps = args.steps_per_epoch or len(tx) // 4
        if steps < 8 or steps > len(tx) // 4:
            raise ValueError("steps_per_epoch must be in [8, training_rows/4]")
        for epoch in range(args.epochs):
            order_cpu = torch.randperm(
                len(tx),
                generator=torch.Generator().manual_seed(args.seed + 100 + epoch),
            )
            order = order_cpu.cuda()
            if epoch > 0 and args.mode == "graph":
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
            development, logits, labels = evaluate(model, vx, vy)
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
                "development": development,
                "evaluation_seconds": evaluation_seconds,
                "training_peak_bytes": peak,
                "state": hashes(state),
                "gradients": hashes(gradients),
                "order_sha256": tensor_sha256(order_cpu),
            }
            if args.kind == "cst":
                row["radius"] = radius_statistics(model[2])
            if choose_best(best, development, epoch + 1):
                checkpoint = _checkpoint(
                    model,
                    optimizer,
                    gradients,
                    epoch + 1,
                    updates,
                    development,
                    configuration,
                )
                torch.save(checkpoint, args.output / "best-dev.pt")
                np.savez_compressed(
                    args.output / "best-dev-predictions.npz",
                    logits=logits,
                    labels=labels,
                )
                best = {
                    "epoch": epoch + 1,
                    "updates": updates,
                    "development": development,
                    "checkpoint_sha256": file_sha256(args.output / "best-dev.pt"),
                    "predictions_sha256": file_sha256(
                        args.output / "best-dev-predictions.npz"
                    ),
                    "state": hashes(state),
                    "gradients": hashes(gradients),
                }
                persist(args.output / "best-dev.json", best)
                del checkpoint
            if epoch == args.epochs - 1 or args.retain_epoch_checkpoints:
                torch.save(
                    _checkpoint(
                        model,
                        optimizer,
                        gradients,
                        epoch + 1,
                        updates,
                        development,
                        configuration,
                    ),
                    args.output
                    / (
                        f"epoch-{epoch + 1}.pt"
                        if args.retain_epoch_checkpoints
                        else "final.pt"
                    ),
                )
            epochs.append(row)
            persist(args.output / "epochs.json", epochs)
            persist(args.output / "captures.json", captures)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "epoch",
                            "updates",
                            "mean_step_ms",
                            "train_loss",
                            "train_accuracy",
                            "development",
                        )
                    }
                ),
                flush=True,
            )
            del state, gradients, logits, labels
        torch.cuda.synchronize()
        begin = time.perf_counter()
        program = None
        gc.collect()
        torch.cuda.synchronize()
        preparation_seconds += time.perf_counter() - begin
        extension = None
        if args.kind == "cst":
            path = Path(executor._extension().__file__)
            extension = {"path": str(path), "sha256": file_sha256(path)}
        result = {
            "schema_version": 1,
            "cohort": HIGGS_CONFIRMATION_COHORT,
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
            "development_rows": len(vy),
            "batch_size": 4,
            "validation_partition": {
                "development": [0, dev_rows],
                "confirmation": [DEV_ROWS, DEV_ROWS + CONFIRMATION_ROWS],
            },
            "checkpoint_selection": "minimum development cross-entropy; earliest epoch on exact ties; no confirmation/test selection",
            "dataset_sha256": data_sha,
            "initialization_sha256": file_sha256(args.output / "initialization.pt"),
            "initialization_policy": "public stock radius CST; standard independent nn.Linear dense; common ordinary layers; no gain calibration",
            "initial_activations": initial_activation,
            "initial_development": initial_development,
            "kernel_settings": radius_settings() if args.kind == "cst" else None,
            "optimizer_contract": contract,
            "epochs": epochs,
            "captures": captures,
            "best_development_checkpoint": best,
            "final_checkpoint": f"epoch-{args.epochs}.pt"
            if args.retain_epoch_checkpoints
            else "final.pt",
            "checkpoint_retention": "all epochs plus best-dev"
            if args.retain_epoch_checkpoints
            else "best-dev and final",
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
            "memory_scope": "allocator reset before optimizer; includes bootstrap/capture/restoration, resident training/development data and prior development evaluations; excludes deferred confirmation; not total process memory",
            "timing_scope": "synchronized gather/forward/loss/backward/update/metrics; adoption adds preparation/capture/restoration/retirement; evaluation/checkpoint/hash diagnostics excluded",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cpu_math_threads": 1,
            "TF32": False,
            "extension": extension,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        persist(args.output / "learning.json", result)
    return result


def evaluate_checkpoint(args, checkpoint, output):
    """Score a frozen checkpoint on the confirmation partition, never select it."""
    _configure()
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"confirmation output already exists: {output}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data_sha = file_sha256(args.data)
    validate_checkpoint(state["configuration"], args, data_sha)
    selected = json.loads((checkpoint.parent / "best-dev.json").read_text())
    checkpoint_sha = file_sha256(checkpoint)
    if selected["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("confirmation requires the frozen best-development checkpoint")
    output.mkdir(parents=True)
    model = make_model(args.width, args.divisor, args.seed, args.kind)
    model.load_state_dict(state["model"])
    x, y = load_partition(args.data, "confirmation")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model = model.cuda()
        begin = time.perf_counter()
        metrics, logits, labels = evaluate(model, x.cuda(), y.cuda())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
    np.savez_compressed(output / "predictions.npz", logits=logits, labels=labels)
    result = {
        "schema_version": 1,
        "cohort": HIGGS_CONFIRMATION_COHORT,
        "kind": args.kind,
        "width": args.width,
        "divisor": args.divisor,
        "alpha": 1 / args.divisor,
        "seed": args.seed,
        "lr": state["configuration"]["lr"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_configuration": state["configuration"],
        "checkpoint_epoch": state["epoch"],
        "checkpoint_updates": state["updates"],
        "checkpoint_development": state["development"],
        "dataset_sha256": data_sha,
        "validation_partition": [DEV_ROWS, DEV_ROWS + CONFIRMATION_ROWS],
        "confirmation": metrics,
        "predictions_sha256": file_sha256(output / "predictions.npz"),
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "evaluation_seconds": seconds,
        "batch_size": 4,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "TF32": False,
        "selection_scope": "checkpoint selected exclusively on development loss before this evaluation",
        "test_evaluated": False,
    }
    persist(output / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument(
        "--width", type=int, choices=HIGGS_CONFIRMATION_WIDTHS, default=1024
    )
    parser.add_argument(
        "--divisor", type=int, choices=HIGGS_CONFIRMATION_DIVISORS, default=100
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--development-rows", type=int, default=DEV_ROWS)
    parser.add_argument("--retain-epoch-checkpoints", action="store_true")
    parser.add_argument("--evaluate-confirmation", action="store_true")
    parser.add_argument("--evaluate-only-checkpoint", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    if args.evaluate_only_checkpoint is not None:
        if args.evaluation_output is None:
            parser.error("--evaluate-only-checkpoint requires --evaluation-output")
        evaluate_checkpoint(args, args.evaluate_only_checkpoint, args.evaluation_output)
    else:
        run(args)
        if args.evaluate_confirmation:
            evaluate_checkpoint(
                args,
                args.output / "best-dev.pt",
                args.evaluation_output or args.output / "confirmation",
            )


if __name__ == "__main__":
    main()
