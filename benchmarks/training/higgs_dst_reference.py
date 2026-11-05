"""Matched-budget static/RigL-style HIGGS accuracy reference.

Dense storage and dense matrix products intentionally make no sparse speed or
memory claim. Only the middle layer is masked; all three layers learn. The
confirmation half of validation and the official test set are never evaluated.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks.training.cell_mass_mnist import tensor_sha256
from benchmarks.training.full_step_graph import exact, live, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.higgs_confirmation import (
    _configure,
    atom_count,
    choose_best,
    development_rows,
    evaluate,
    file_sha256,
    load_partition,
)
from benchmarks.training.higgs_density import (
    initial_activations,
    persist,
    save_initialization,
)

COHORT = "higgs_dst_reference_v1"


class MaskedLinear(nn.Module):
    """Exact row-balanced active budget, with an explicit dense growth gradient.

    Inactive stored weights/moments are always zero. Linear's full dW supplies
    RigL's dormant-coordinate scores; masking dW before AdamW gives precisely
    the active derivatives of linear(x, weight * mask). This also avoids
    backward hooks or dynamic allocations in the optional CUDA Graph.
    """

    def __init__(self, width, active_count, generator):
        super().__init__()
        if not 1 <= active_count // width <= width or active_count > width * width:
            raise ValueError("budget must give every row at least one active entry")
        per_row = torch.full((width,), active_count // width, dtype=torch.int64)
        per_row[: active_count % width] += 1
        mask = torch.zeros(width, width, dtype=torch.bool)
        weight = torch.zeros(width, width)
        for row, count in enumerate(per_row.tolist()):
            columns = torch.randperm(width, generator=generator)[:count]
            mask[row, columns] = True
            weight[row, columns] = torch.empty(count).uniform_(
                -1 / math.sqrt(count), 1 / math.sqrt(count), generator=generator
            )
        self.weight = nn.Parameter(weight)
        self.register_buffer("mask", mask)
        self.register_buffer("row_active_counts", per_row)
        self.register_buffer(
            "dense_gradient", torch.zeros_like(weight), persistent=False
        )

    def forward(self, x):
        return nn.functional.linear(x, self.weight)

    @torch.no_grad()
    def prepare_gradient(self):
        self.dense_gradient.copy_(self.weight.grad)
        self.weight.grad.mul_(self.mask)


def make_model(width, divisor, seed):
    count = 4 * atom_count(width, divisor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = nn.Linear(28, width)
        torch.manual_seed(seed + 2)
        last = nn.Linear(width, 2)
        middle = MaskedLinear(width, count, torch.Generator().manual_seed(seed + 3))
    return nn.Sequential(first, nn.ReLU(), middle, nn.ReLU(), last)


def make_optimizer(model, lr, capturable=False):
    ordinary = [p for index in (0, 4) for p in model[index].parameters()]
    return torch.optim.AdamW(
        [{"params": ordinary}, {"params": [model[2].weight]}],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
        capturable=capturable,
    )


def rewiring_fraction(step, total_steps, initial_fraction=0.3, end_fraction=0.75):
    end = int(total_steps * end_fraction)
    if not 0 <= initial_fraction <= 1 or not 0 < end_fraction <= 1 or end < 1:
        raise ValueError("invalid cosine rewiring schedule")
    if step >= end:
        return 0.0
    return initial_fraction * (1 + math.cos(math.pi * step / end)) / 2


@torch.no_grad()
def rewire(linear, optimizer, fraction, step, optimizer_updates=None):
    """Row-conserving magnitude prune / dense-gradient grow, with stable ties.

    Newly grown positions include just-pruned positions. All pruned/regrown
    entries start at zero with zero Adam moments; only true survivors retain
    values and moments. A global magnitude threshold chooses pruning; each row
    retains at least one old position and regrows its exact initial budget.
    """
    if not 0 < fraction < 1:
        raise ValueError("a topology event requires a fraction strictly within (0,1)")
    mask, weight = linear.mask, linear.weight
    old_mask = mask.clone()
    before_hash = tensor_sha256(old_mask)
    magnitude = weight.abs()
    count = int(old_mask.sum())
    prune_count = max(1, int(fraction * count))
    threshold = magnitude[old_mask].kthvalue(prune_count).values
    keep = old_mask & (magnitude > threshold)
    empty = ~keep.any(dim=1)
    rescue = magnitude.masked_fill(~old_mask, -1).argmax(dim=1)
    rows = empty.nonzero(as_tuple=True)[0]
    keep[rows, rescue[rows]] = True
    missing = linear.row_active_counts - keep.sum(1)
    candidates = linear.dense_gradient.abs().masked_fill(keep, -torch.inf)
    # Stable sorting makes equal gradient scores choose the lower column.
    order = candidates.argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        1, order, torch.arange(mask.shape[1], device=mask.device).expand_as(order)
    )
    grown = rank < missing[:, None]
    new_mask = keep | grown
    moments = optimizer.state[weight]
    old_moments = {k: moments[k].clone() for k in ("exp_avg", "exp_avg_sq")}
    weight.mul_(keep)
    mask.copy_(new_mask)
    weight.grad.mul_(keep)
    for name in old_moments:
        moments[name].mul_(keep)
    checks = {
        "active_count_conserved": int(mask.sum()) == count,
        "row_counts_conserved": torch.equal(mask.sum(1), linear.row_active_counts),
        "inactive_weights_zero": bool((weight[~mask] == 0).all()),
        "grown_weights_zero": bool((weight[grown] == 0).all()),
        "pruned_and_grown_moments_zero": all(
            bool((moments[k][~keep] == 0).all()) for k in old_moments
        ),
        "survivor_moments_preserved": all(
            torch.equal(moments[k][keep], old_moments[k][keep]) for k in old_moments
        ),
        "optimizer_clock_unchanged": float(moments["step"])
        == (step if optimizer_updates is None else optimizer_updates),
    }
    assert all(checks.values()), checks
    return {
        "step": step,
        "optimizer_updates": step if optimizer_updates is None else optimizer_updates,
        "drop_fraction": fraction,
        "requested_prune_count": prune_count,
        "pruned_before_growth": int((old_mask & ~keep).sum()),
        "grown_slots": int(grown.sum()),
        "dropped": int((old_mask & ~mask).sum()),
        "grown": int((mask & ~old_mask).sum()),
        "pruned_then_regrown": int((grown & old_mask).sum()),
        "survived": int(keep.sum()),
        "active_count": count,
        "topology_before_sha256": before_hash,
        "topology_after_sha256": tensor_sha256(mask),
        "dense_gradient_sha256": tensor_sha256(linear.dense_gradient),
        "moment_reset_rule": "preserve only pruning survivors; reset all pruned/grown coordinates including immediate regrowth",
        "moment_hashes_after": {k: tensor_sha256(moments[k]) for k in old_moments},
        "checks": checks,
    }


def training_step(model, optimizer, tx, ty, indices, totals, apply_update=True):
    optimizer.zero_grad(set_to_none=False)
    labels = ty[indices]
    logits = model(tx[indices])
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    model[2].prepare_gradient()
    if apply_update:
        optimizer.step()
    with torch.no_grad():
        totals[0].add_(loss * 4)
        totals[1].add_((logits.argmax(1) == labels).sum())


class PreparedStep:
    """Fixed shape pure-Torch graph; topology mutates existing tensor storage."""

    def __init__(self, model, optimizer, tx, ty, totals, apply_update=True):
        self.model, self.optimizer = model, optimizer
        self.indices = torch.arange(4, device=tx.device)
        self.totals = totals
        records = list(live(model, optimizer).values())
        records += [p.grad for p in model.parameters()]
        records += [model[2].dense_gradient, *totals]
        saved = [value.clone() for value in records]
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=torch.cuda.current_stream()):
            training_step(model, optimizer, tx, ty, self.indices, totals, apply_update)
        with torch.no_grad():
            for value, original in zip(records, saved):
                value.copy_(original)
        assert all(
            torch.equal(value, original) for value, original in zip(records, saved)
        )

    def step(self, indices):
        self.indices.copy_(indices)
        self.graph.replay()


def checkpoint(model, optimizer, gradients, row, configuration, events):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "gradients": gradients,
        "dense_growth_gradient": model[2].dense_gradient.detach().cpu().clone(),
        "epoch": row["epoch"],
        "updates": row["updates"],
        "minibatch_step": row["minibatch_step"],
        "data_examples_seen": row["data_examples_seen"],
        "actual_optimizer_updates": row["actual_optimizer_updates"],
        "development": row["development"],
        "configuration": configuration,
        "topology_events": events.copy(),
    }


def run(args):
    _configure()
    args.kind = args.kind.removeprefix("dst-")
    if args.kind not in ("static", "rigl"):
        raise ValueError("kind must be static or rigl")
    if not 0 < args.drop_fraction < 1 or not 0 < args.end_fraction <= 1:
        raise ValueError("invalid rewiring schedule")
    dev_rows = development_rows(args)
    if (
        args.epochs < 1
        or args.update_frequency < 2
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("positive epochs/LR and update frequency >=2 required")
    args.output.mkdir(parents=True, exist_ok=False)
    data_sha = file_sha256(args.data)
    with np.load(args.data, allow_pickle=False) as payload:
        tx = torch.from_numpy(payload["x_train"].copy())
        ty = torch.from_numpy(payload["y_train"].copy())
    if (
        tx.shape != (100000, 28)
        or ty.shape != (100000,)
        or tx.dtype != torch.float32
        or ty.dtype != torch.int64
    ):
        raise ValueError("expected FP32/INT64 100k HIGGS training payload")
    vx, vy = load_partition(args.data, "development")
    vx, vy = vx[:dev_rows], vy[:dev_rows]
    model = make_model(args.width, args.divisor, args.seed)
    configuration = {
        "cohort": COHORT,
        "kind": args.kind,
        "width": args.width,
        "divisor": args.divisor,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "dataset_sha256": data_sha,
    }
    saved_args = vars(args) | {"data": str(args.data), "output": str(args.output)}
    save_initialization(model, saved_args, args.output / "initialization.pt")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx, ty, vx, vy = (value.cuda() for value in (tx, ty, vx, vy))
        model = model.cuda()
        initial_activation = initial_activations(model, tx)
        initial_development, _, _ = evaluate(model, vx, vy)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer = make_optimizer(model, args.lr, capturable=True)
        totals = (
            tx.new_zeros(()),
            torch.zeros((), device=tx.device, dtype=torch.int64),
        )
        steps = args.steps_per_epoch or len(tx) // 4
        if not 8 <= steps <= 25000:
            raise ValueError("step budget must be 8..25000")
        total_steps = steps * args.epochs
        end_step = int(total_steps * args.end_fraction)
        events, epochs, captures = [], [], []
        best, program, topology_program, updates, optimizer_updates, replays = (
            None,
            None,
            None,
            0,
            0,
            0,
        )
        training_seconds, preparation_seconds = 0.0, 0.0
        for epoch in range(1, args.epochs + 1):
            order_cpu = torch.randperm(
                len(tx), generator=torch.Generator().manual_seed(args.seed + 99 + epoch)
            )
            order = order_cpu.cuda()
            for value in totals:
                value.zero_()
            torch.cuda.synchronize()
            epoch_begin = time.perf_counter()
            epoch_preparation = 0.0
            for index in range(steps):
                indices = order[4 * index : 4 * index + 4]
                if updates == 8 and args.mode == "graph":
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    before = snapshot(model, optimizer)
                    program = PreparedStep(model, optimizer, tx, ty, totals)
                    topology_program = PreparedStep(
                        model, optimizer, tx, ty, totals, apply_update=False
                    )
                    torch.cuda.synchronize()
                    restored = exact(before, snapshot(model, optimizer))
                    assert restored["all_byte_equal"]
                    seconds = time.perf_counter() - begin
                    epoch_preparation += seconds
                    captures.append(
                        {
                            "at_update": updates,
                            "seconds": seconds,
                            "restoration": restored,
                        }
                    )
                topology_step = (
                    args.kind == "rigl"
                    and (updates + 1) % args.update_frequency == 0
                    and updates + 1 < end_step
                )
                if program is None:
                    training_step(
                        model,
                        optimizer,
                        tx,
                        ty,
                        indices,
                        totals,
                        apply_update=not topology_step,
                    )
                else:
                    (topology_program if topology_step else program).step(indices)
                    replays += 1
                updates += 1
                optimizer_updates += not topology_step
                if topology_step:
                    fraction = rewiring_fraction(
                        updates, total_steps, args.drop_fraction, args.end_fraction
                    )
                    events.append(
                        rewire(
                            model[2], optimizer, fraction, updates, optimizer_updates
                        )
                    )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - epoch_begin - epoch_preparation
            training_seconds += seconds
            preparation_seconds += epoch_preparation
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
                float(value["step"]) == optimizer_updates
                for value in optimizer.state.values()
            )
            assert int(model[2].mask.sum()) == 4 * atom_count(args.width, args.divisor)
            assert bool((model[2].weight[~model[2].mask] == 0).all())
            peak = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            begin = time.perf_counter()
            development, logits, labels = evaluate(model, vx, vy)
            torch.cuda.synchronize()
            assert exact(state, snapshot(model, optimizer))["all_byte_equal"]
            row = {
                "epoch": epoch,
                "updates": updates,
                "minibatch_step": updates,
                "data_examples_seen": 4 * updates,
                "actual_optimizer_updates": optimizer_updates,
                "replays": replays,
                "training_seconds": seconds,
                "mean_step_ms": seconds * 1000 / steps,
                "train_loss": float(totals[0]) / (4 * steps),
                "train_accuracy": int(totals[1]) / (4 * steps),
                "development": development,
                "evaluation_seconds": time.perf_counter() - begin,
                "training_peak_bytes": peak,
                "state": hashes(state),
                "gradients": hashes(gradients),
                "dense_growth_gradient_sha256": tensor_sha256(model[2].dense_gradient),
                "order_sha256": tensor_sha256(order_cpu),
                "topology_sha256": tensor_sha256(model[2].mask),
                "topology_event_count": len(events),
            }
            if choose_best(best, development, epoch):
                torch.save(
                    checkpoint(model, optimizer, gradients, row, configuration, events),
                    args.output / "best-dev.pt",
                )
                np.savez_compressed(
                    args.output / "best-dev-predictions.npz",
                    logits=logits,
                    labels=labels,
                )
                best = {
                    "epoch": epoch,
                    "updates": updates,
                    "minibatch_step": updates,
                    "data_examples_seen": 4 * updates,
                    "actual_optimizer_updates": optimizer_updates,
                    "development": development,
                    "state": row["state"],
                    "gradients": row["gradients"],
                    "checkpoint_sha256": file_sha256(args.output / "best-dev.pt"),
                    "predictions_sha256": file_sha256(
                        args.output / "best-dev-predictions.npz"
                    ),
                }
                persist(args.output / "best-dev.json", best)
            if epoch == args.epochs or args.retain_epoch_checkpoints:
                path = args.output / (
                    f"epoch-{epoch}.pt" if args.retain_epoch_checkpoints else "final.pt"
                )
                torch.save(
                    checkpoint(model, optimizer, gradients, row, configuration, events),
                    path,
                )
            epochs.append(row)
            persist(args.output / "epochs.json", epochs)
            persist(args.output / "topology-events.json", events)
            print(row["epoch"], row["development"], flush=True)
        torch.cuda.synchronize()
        result = configuration | {
            "schema_version": 1,
            "alpha": 1 / args.divisor,
            "atom_count_matched": atom_count(args.width, args.divisor),
            "architecture": f"28-{args.width}-{args.width}-2; ReLU; bias-free masked middle",
            "all_layers_trainable": True,
            "trainable_parameters": 4 * atom_count(args.width, args.divisor)
            + 31 * args.width
            + 2,
            "stored_parameters": sum(p.numel() for p in model.parameters()),
            "active_middle_weights": int(model[2].mask.sum()),
            "mode": args.mode,
            "batch_size": 4,
            "training_rows": len(tx),
            "development_rows": dev_rows,
            "validation_partition": {
                "development": [0, dev_rows],
                "confirmation": [50000, 100000],
            },
            "initialization_sha256": file_sha256(args.output / "initialization.pt"),
            "initialization_policy": "row-balanced exact4A active entries; per-row fan-in uniform; common CST ordinary-layer RNG streams",
            "initial_activations": initial_activation,
            "initial_development": initial_development,
            "optimizer_contract": {
                "algorithm": "AdamW",
                "lr": args.lr,
                "weight_decay_all_parameters": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
                "capturable": True,
            },
            "rewiring": {
                "algorithm": args.kind,
                "update_frequency": args.update_frequency,
                "initial_drop_fraction": args.drop_fraction,
                "end_fraction": args.end_fraction,
                "end_step_exclusive": end_step,
                "cosine_horizon": "rewiring cutoff",
                "growth": "largest absolute full dense dW per row among non-survivors; stable column ties",
                "reset": "zero pruned/grown values and both moments, including immediately regrown entries",
                "optimizer_update_order": "topology event replaces all-layer AdamW; same minibatch scores at pre-update weights; scalar Adam clocks unchanged on structural batches",
                "event_bounds": "positive minibatch step divisible by update frequency and strictly below floor(end_fraction * total_minibatches)",
                "variant": "row-balanced HIGGS RigL-style; all pruned/grown values and moments reset, including immediate regrowth",
            },
            "epochs": epochs,
            "captures": captures,
            "topology_events": events,
            "best_development_checkpoint": best,
            "final_checkpoint": f"epoch-{args.epochs}.pt"
            if args.retain_epoch_checkpoints
            else "final.pt",
            "checkpoint_selection": "minimum development CE; earliest exact tie",
            "checkpoint_retention": "all epochs plus best-dev"
            if args.retain_epoch_checkpoints
            else "best-dev and final",
            "total_updates": updates,
            "minibatch_step": updates,
            "data_examples_seen": 4 * updates,
            "actual_optimizer_updates": optimizer_updates,
            "curve_x_axis": "minibatch_step or data_examples_seen; optimizer clocks differ on RigL topology batches",
            "graph_replays": replays,
            "training_seconds": training_seconds,
            "preparation_seconds": preparation_seconds,
            "adoption_seconds": training_seconds + preparation_seconds,
            "training_peak_allocated_bytes": max(
                row["training_peak_bytes"]["allocated"] for row in epochs
            ),
            "training_peak_reserved_bytes": max(
                row["training_peak_bytes"]["reserved"] for row in epochs
            ),
            "memory_scope": "dense masked reference; includes dense weights/moments/full growth-gradient scratch and resident train/dev data; allocator peak includes capture",
            "timing_scope": "dense reference complete updates plus topology events including event audit hashes; no sparse implementation efficiency claim; dev evaluation/checkpoints/per-epoch state hashes excluded",
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cpu_math_threads": 1,
            "TF32": False,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        persist(args.output / "learning.json", result)
    return result


def vjp_fixture(width=17, active_count=51, device="cpu"):
    """Independent masked-forward VJP and full dense grow-score oracle."""
    layer = MaskedLinear(width, active_count, torch.Generator().manual_seed(123)).to(
        device
    )
    generator = torch.Generator().manual_seed(456)
    x = torch.randn(4, width, generator=generator).to(device).requires_grad_()
    dy = torch.randn(4, width, generator=generator).to(device)
    y = layer(x)
    y.backward(dy)
    layer.prepare_gradient()
    oracle_x = x.detach().clone().requires_grad_()
    oracle_weight = layer.weight.detach().clone().requires_grad_()
    expected_y = nn.functional.linear(oracle_x, oracle_weight * layer.mask)
    expected_dx, expected_dw = torch.autograd.grad(
        expected_y, (oracle_x, oracle_weight), dy
    )
    actual = (y.detach(), x.grad, layer.weight.grad, layer.dense_gradient)
    expected = (expected_y.detach(), expected_dx, expected_dw, dy.T @ x.detach())
    checks = {
        name: torch.equal(left, right)
        for name, left, right in zip(
            ("Y", "dX", "masked_dW", "dense_growth_dW"), actual, expected
        )
    }
    assert all(checks.values()), checks
    return {
        "width": width,
        "active_count": active_count,
        "actual": tuple(value.cpu() for value in actual),
        "expected": tuple(value.cpu() for value in expected),
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualification_trajectory(width, divisor, kind, mode, output=None):
    """Two epochs, with events before/after capture and across epoch boundary."""
    _configure()
    model = make_model(width, divisor, 17).cuda()
    optimizer = make_optimizer(model, 0.0003, capturable=True)
    generator = torch.Generator().manual_seed(999)
    tx = torch.randn(64, 28, generator=generator).cuda()
    ty = torch.randint(0, 2, (64,), generator=generator).cuda()
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    events, states, orders = [], [], []
    program, topology_program, optimizer_updates = None, None, 0
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        torch.save(
            {"model": copy.deepcopy(model.state_dict())}, output / "initialization.pt"
        )
    for epoch in (1, 2):
        order = torch.randperm(
            64, generator=torch.Generator().manual_seed(116 + epoch)
        ).cuda()
        orders.append(tensor_sha256(order))
        for value in totals:
            value.zero_()
        for index in range(16):
            step = (epoch - 1) * 16 + index + 1
            if step == 9 and mode == "graph":
                program = PreparedStep(model, optimizer, tx, ty, totals)
                topology_program = PreparedStep(
                    model, optimizer, tx, ty, totals, apply_update=False
                )
            topology_step = kind == "rigl" and step % 4 == 0 and step < 24
            indices = order[4 * index : 4 * index + 4]
            if program is None:
                training_step(
                    model, optimizer, tx, ty, indices, totals, not topology_step
                )
            else:
                (topology_program if topology_step else program).step(indices)
            optimizer_updates += not topology_step
            if topology_step:
                events.append(
                    rewire(
                        model[2],
                        optimizer,
                        rewiring_fraction(step, 32),
                        step,
                        optimizer_updates,
                    )
                )
        values = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "gradients": {
                name: p.grad.detach().clone() for name, p in model.named_parameters()
            },
            "dense_gradient": model[2].dense_gradient.clone(),
            "totals": tuple(value.clone() for value in totals),
        }
        from benchmarks.training.higgs_verify import tensor_fields

        values = {
            name: value.detach().cpu().clone() for name, value in tensor_fields(values)
        }
        states.append(values)
        if output is not None:
            torch.save(values, output / f"epoch-{epoch}.pt")
    result = {
        "kind": kind,
        "width": width,
        "divisor": divisor,
        "mode": mode,
        "states": states,
        "events": events,
        "data_orders": orders,
        "minibatch_step": 32,
        "data_examples_seen": 128,
        "actual_optimizer_updates": optimizer_updates,
    }
    if output is not None:
        persist(
            output / "trajectory.json",
            {k: value for k, value in result.items() if k != "states"}
            | {"state_hashes": [hashes(state) for state in states]},
        )
    return result


def qualify(output):
    _configure()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "cohort": COHORT,
        "status": "running",
        "comparisons": [],
        "oracles": [],
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }
    persist(output / "qualification.json", record)
    try:
        cpu_oracle = vjp_fixture()
        torch.save(cpu_oracle, output / "cpu-vjp.pt")
        record["oracles"].append(
            {
                k: value
                for k, value in cpu_oracle.items()
                if k not in ("actual", "expected")
            }
            | {"file": "cpu-vjp.pt", "sha256": file_sha256(output / "cpu-vjp.pt")}
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA qualification requires the selected GPU")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for width in (1024, 2048):
                for divisor in (1000, 500, 200, 100):
                    oracle = vjp_fixture(width, 4 * atom_count(width, divisor), "cuda")
                    name = f"N{width}-D{divisor}-vjp.pt"
                    torch.save(oracle, output / name)
                    record["oracles"].append(
                        {
                            k: value
                            for k, value in oracle.items()
                            if k not in ("actual", "expected")
                        }
                        | {"file": name, "sha256": file_sha256(output / name)}
                    )
                for divisor in (1000, 100):
                    for kind in ("static", "rigl"):
                        label = f"N{width}-D{divisor}-{kind}"
                        eager = qualification_trajectory(
                            width, divisor, kind, "eager", output / (label + "-eager")
                        )
                        graph = qualification_trajectory(
                            width, divisor, kind, "graph", output / (label + "-graph")
                        )
                        equality = [
                            exact(left, right)
                            for left, right in zip(eager["states"], graph["states"])
                        ]
                        checks = {
                            "all_epoch_tensor_bytes": all(
                                item["all_byte_equal"] for item in equality
                            ),
                            "topology_event_bytes_and_checks": eager["events"]
                            == graph["events"],
                            "same_data_orders": eager["data_orders"]
                            == graph["data_orders"],
                            "optimizer_update_count": eager["actual_optimizer_updates"]
                            == graph["actual_optimizer_updates"]
                            == (27 if kind == "rigl" else 32),
                            "expected_event_steps": [
                                event["step"] for event in eager["events"]
                            ]
                            == ([4, 8, 12, 16, 20] if kind == "rigl" else []),
                        }
                        assert all(checks.values()), checks
                        record["comparisons"].append(
                            {
                                "label": label,
                                "checks": checks,
                                "epoch_equality": equality,
                                "pass": all(checks.values()),
                            }
                        )
                        persist(output / "qualification.json", record)
        record.update(
            status="passed",
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            TF32=False,
        )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist(output / "qualification.json", record)
        raise
    persist(output / "qualification.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("static", "rigl", "dst-static", "dst-rigl"))
    parser.add_argument("--width", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument(
        "--divisor", type=int, choices=(1000, 500, 200, 100), default=1000
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--development-rows", type=int, default=50000)
    parser.add_argument("--retain-epoch-checkpoints", action="store_true")
    parser.add_argument("--update-frequency", type=int, default=250)
    parser.add_argument("--drop-fraction", type=float, default=0.3)
    parser.add_argument("--end-fraction", type=float, default=0.75)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.qualify_only:
        qualify(args.output)
    else:
        if args.data is None or args.kind is None:
            parser.error("training requires --data and --kind")
        run(args)


if __name__ == "__main__":
    main()
