#!/usr/bin/env python3
"""CONV-GROW-4 protocol on a continuous Gaussian ``CSTLinear`` filter.

This is a protocol reproduction, not a claim that the representation is the
same as CONV-GROW-4's per-offset rank-one blocks. The topology, MNIST split,
batch tape, 8->16 target convolution, K ladder, growth times, optimizer, and
600-update horizon are retained. The target filter is instead one ``CSTConv2d``
whose patch map is a continuous ``CSTLinear`` over ``(channel, dy, dx)`` input
coordinates and one-dimensional output-channel coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from experiments.e2e_mnist import load_mnist
except ModuleNotFoundError:  # Direct ``python experiments/e2e_mnist_conv.py``.
    from e2e_mnist import load_mnist
from torchcst.compute import (
    BackwardContext,
    CSTConv2d,
    CSTLinear,
    Observation,
    conv2d_neuron_coordinates,
)
from torchcst.optim import OptimizerStateFollower, parameter_groups
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


N_IN = 8
N_OUT = 16
KERNEL_SIZE = 3
K0 = 4
K_FINAL = 36
BATCH_SIZE = 128
STEPS = 600
BIRTH_STEPS = (100, 150, 200, 250, 300, 350)
SEEDS = (0, 1, 2)
SIGMA_IN = 0.30
SIGMA_OUT = 0.30
INIT_WEIGHT_STD = 0.05
CANDIDATE_POOL = 256
ARMS = (
    "dense",
    "cst-static-K9",
    "cst-static-K18",
    "cst-static-K36",
    "cst-gradient-grow",
    "cst-rand-grow",
    "cst-jump",
)
REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "cst" / "data" / "MNIST" / "raw"
DEFAULT_OUTPUT = REPO / "results" / "e2e_mnist_conv.json"


@dataclass(frozen=True)
class Protocol:
    steps: int
    batch_size: int
    train_n: int
    eval_n: int
    birth_steps: tuple[int, ...]
    arms: tuple[str, ...]


FULL_PROTOCOL = Protocol(STEPS, BATCH_SIZE, 8192, 2048, BIRTH_STEPS, ARMS)
SMOKE_PROTOCOL = Protocol(
    20,
    32,
    512,
    128,
    (5, 8, 11, 14, 17, 20),
    ("dense", "cst-static-K9", "cst-gradient-grow", "cst-rand-grow"),
)


class ConvNet(nn.Module):
    def __init__(
        self,
        target: nn.Module,
        *,
        stem: nn.Conv2d | None = None,
        head: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, N_IN, 3, padding=1) if stem is None else stem
        self.target = target
        self.head = nn.Linear(N_OUT, 10) if head is None else head

    def forward(self, x: Tensor) -> Tensor:
        x = F.max_pool2d(F.gelu(self.stem(x)), 2)
        x = F.gelu(self.target(x)).mean(dim=(2, 3))
        return self.head(x)


def birth_counts(birth_steps: Sequence[int]) -> tuple[int, ...]:
    quotient, remainder = divmod(K_FINAL - K0, len(birth_steps))
    return tuple(quotient + (index < remainder) for index in range(len(birth_steps)))


def active_k(step: int, birth_steps: Sequence[int]) -> int:
    value = K0
    for event_step, count in zip(birth_steps, birth_counts(birth_steps)):
        if step >= event_step:
            value += count
    return value


def matched_jump_step(steps: int, birth_steps: Sequence[int]) -> int:
    progressive = sum(active_k(step, birth_steps) for step in range(1, steps + 1))
    candidates = []
    for jump in range(1, steps + 1):
        total = K0 * (jump - 1) + K_FINAL * (steps - jump + 1)
        candidates.append((abs(total - progressive), jump))
    return min(candidates)[1]


def make_batch_tape(n_train: int, protocol: Protocol, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(91_000 + seed)
    return torch.randint(
        n_train,
        (protocol.steps, protocol.batch_size),
        generator=generator,
    )


def tensor_hash(tensor: Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def load_protocol_data(
    data_dir: Path, protocol: Protocol
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    images, labels, _, _ = load_mnist(
        data_dir,
        train_limit=60_000,
        test_limit=1,
    )
    images = F.avg_pool2d(images.reshape(-1, 1, 28, 28), 2)
    images = (images - 0.1307) / 0.3081
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(811))
    if protocol.train_n + protocol.eval_n > len(images):
        raise ValueError("protocol split exceeds the MNIST training set")
    train_ids = order[: protocol.train_n]
    eval_ids = order[protocol.train_n : protocol.train_n + protocol.eval_n]
    return images[train_ids], labels[train_ids], images[eval_ids], labels[eval_ids]


def _bank(seed: int, count: int) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(43_000 + seed)
    source = torch.rand(count, 3, generator=generator)
    target = torch.rand(count, 1, generator=generator)
    weights = torch.randn(count, generator=generator) * INIT_WEIGHT_STD
    return source, target, weights


def _initial_k(arm: str) -> int:
    if arm.startswith("cst-static-K"):
        return int(arm.rsplit("K", 1)[1])
    return K0


def build_arm(
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[ConvNet, SynapseStore | None, CSTConv2d | None, torch.optim.Optimizer]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(42_000 + seed)
    if arm == "dense":
        # Preserve CONV-GROW-4 DenseMNISTCNN's exact RNG consumption order.
        stem = nn.Conv2d(1, N_IN, 3, padding=1)
        target_module = nn.Conv2d(N_IN, N_OUT, KERNEL_SIZE, padding=1)
        head = nn.Linear(N_OUT, 10)
        model = ConvNet(target_module, stem=stem, head=head).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        return model, None, None, optimizer

    stem = nn.Conv2d(1, N_IN, 3, padding=1)
    head = nn.Linear(N_OUT, 10)
    initial_k = _initial_k(arm)
    capacity = initial_k if arm.startswith("cst-static") else K_FINAL
    source, target, weights = _bank(seed, initial_k)
    store = SynapseStore(
        "mnist-conv",
        3,
        1,
        capacity,
        spec=RepresentationSpec.continuous(3, 1),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                source,
                target,
                weights,
                torch.arange(initial_k, dtype=torch.int64),
            )
        ]
    )
    input_mu, output_mu = conv2d_neuron_coordinates(N_IN, N_OUT, KERNEL_SIZE)
    inputs = NeuronStore(
        "mnist-conv-in",
        len(input_mu),
        mu=input_mu,
        initial_live=len(input_mu),
    )
    outputs = NeuronStore(
        "mnist-conv-out",
        len(output_mu),
        mu=output_mu,
        initial_live=len(output_mu),
    )
    inputs.gate.requires_grad_(False)
    outputs.gate.requires_grad_(False)
    linear = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianKernel(SIGMA_IN, learnable=False),
        GaussianKernel(SIGMA_OUT, learnable=False),
    )
    conv = CSTConv2d(linear, N_IN, KERNEL_SIZE, padding=1)
    model = ConvNet(conv, stem=stem, head=head).to(device)
    groups = parameter_groups(
        model,
        amplitude_lr=2.0e-3,
        coordinate_lr=2.0e-4,
        default_lr=2.0e-3,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=1.0e-4)
    follower = OptimizerStateFollower(optimizer, (store.s, store.t, store.w))
    store.followers().subscribe(follower)
    return model, store, conv, optimizer


def _sample_candidates(
    store: SynapseStore,
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    source = store.spec.domain_in.sample(count, generator).to(store.s)
    target = store.spec.domain_out.sample(count, generator).to(store.t)
    return source, target


@torch.no_grad()
def candidate_gradients(
    conv: CSTConv2d,
    observation: Observation,
    source: Tensor,
    target: Tensor,
) -> Tensor:
    linear = conv.linear
    x = observation.x.reshape(-1, linear.in_features)
    g_out = observation.g_out.reshape(-1, linear.out_features)
    source = source.to(device=x.device, dtype=x.dtype)
    target = target.to(device=x.device, dtype=x.dtype)
    input_mu = linear.in_neurons.mu.to(source)
    output_mu = linear.out_neurons.mu.to(target)
    k_in = linear.kernel_in(input_mu, source)
    k_out = linear.kernel_out(output_mu, target)
    return ((x @ k_in) * (g_out @ k_out)).sum(dim=0)


def _finalize_observation(
    conv: CSTConv2d,
    context: BackwardContext,
    loss: Tensor,
) -> Observation:
    try:
        loss.backward()
        context.observe_microbatch()
        observations = context.finalize()
    finally:
        conv.set_backward_context(None)
    if len(observations) != 1:
        raise RuntimeError("target convolution must emit exactly one observation")
    return observations[0]


@torch.no_grad()
def apply_birth(
    store: SynapseStore,
    optimizer: torch.optim.Optimizer,
    source: Tensor,
    target: Tensor,
    gradients: Tensor,
    *,
    lineage_start: int,
) -> tuple[int, ...]:
    count = len(source)
    before = set(store.live_ids().tolist())
    store.apply(
        [
            SynapseBirth(
                store.site,
                source,
                target,
                store.w.new_zeros(count),
                torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
            )
        ]
    )
    born_ids = torch.tensor(
        sorted(set(store.live_ids().tolist()).difference(before)),
        dtype=torch.int64,
    )
    slots = store._slots.slots_of(born_ids).to(store.w.device)
    if store.w.grad is None:
        store.w.grad = torch.zeros_like(store.w)
    store.w.grad.index_copy_(0, slots, gradients.to(store.w.grad))
    store.reconcile_optimizer_state(optimizer, slots.cpu())
    return tuple(int(value) for value in born_ids.tolist())


def _event_count(step: int, protocol: Protocol, arm: str) -> int:
    if arm in {"cst-gradient-grow", "cst-rand-grow"}:
        for event_step, count in zip(protocol.birth_steps, birth_counts(protocol.birth_steps)):
            if step == event_step:
                return count
    if arm == "cst-jump" and step == matched_jump_step(protocol.steps, protocol.birth_steps):
        return K_FINAL - K0
    return 0


@torch.no_grad()
def evaluate(model: nn.Module, x: Tensor, y: Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss = 0.0
    correct = 0
    for start in range(0, len(x), 256):
        xb = x[start : start + 256].to(device)
        yb = y[start : start + 256].to(device)
        logits = model(xb)
        loss += float(F.cross_entropy(logits, yb, reduction="sum").cpu())
        correct += int((logits.argmax(1) == yb).sum().cpu())
    return loss / len(x), correct / len(x)


def run_arm(
    arm: str,
    seed: int,
    protocol: Protocol,
    data: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> dict[str, Any]:
    train_x, train_y, eval_x, eval_y = data
    tape = make_batch_tape(len(train_x), protocol, seed)
    model, store, conv, optimizer = build_arm(arm, seed, device)
    proposal_rng = torch.Generator().manual_seed(73_000 + seed)
    events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, protocol.steps + 1):
        ids = tape[step - 1]
        xb = train_x[ids].to(device)
        yb = train_y[ids].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        count = _event_count(step, protocol, arm)
        context = None
        if count:
            assert store is not None and conv is not None
            context = BackwardContext(step)
            conv.set_backward_context(context)
        loss = F.cross_entropy(model(xb), yb)
        if count:
            assert store is not None and conv is not None and context is not None
            observation = _finalize_observation(conv, context, loss)
            pool = CANDIDATE_POOL if arm == "cst-gradient-grow" else count
            candidate_s, candidate_t = _sample_candidates(store, pool, proposal_rng)
            scores = candidate_gradients(conv, observation, candidate_s, candidate_t)
            if arm == "cst-gradient-grow":
                selected = torch.argsort(scores.abs(), descending=True, stable=True)[:count]
                candidate_s = candidate_s.index_select(0, selected.to(candidate_s.device))
                candidate_t = candidate_t.index_select(0, selected.to(candidate_t.device))
                scores = scores.index_select(0, selected.to(scores.device))
            old_k = int(store.view().ids.numel())
            born = apply_birth(
                store,
                optimizer,
                candidate_s,
                candidate_t,
                scores,
                lineage_start=step * 1000,
            )
            events.append(
                {
                    "step": step,
                    "old_k": old_k,
                    "new_k": int(store.view().ids.numel()),
                    "born_ids": born,
                    "median_abs_candidate_gradient": float(scores.abs().median().cpu()),
                    "pool": pool,
                }
            )
        else:
            loss.backward()
        optimizer.step()
        if store is not None:
            store.retract_coordinates(optimizer)
        if step == 1 or step % 25 == 0 or count:
            trajectory.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "k": None if store is None else int(store.view().ids.numel()),
                }
            )
    wall = time.perf_counter() - started
    train_loss, train_accuracy = evaluate(model, train_x, train_y, device)
    eval_loss, eval_accuracy = evaluate(model, eval_x, eval_y, device)
    active_k_value = None if store is None else int(store.view().ids.numel())
    active_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if store is not None:
        active_parameters -= 5 * (store.capacity - active_k_value)
    return {
        "arm": arm,
        "seed": seed,
        "steps": protocol.steps,
        "batch_tape_hash": tensor_hash(tape),
        "final_k": active_k_value,
        "active_parameters": active_parameters,
        "stored_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "events": events,
        "trajectory": trajectory,
        "wall_seconds": wall,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(
    protocol: Protocol,
    seeds: Sequence[int],
    data_dir: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    data = load_protocol_data(data_dir, protocol)
    payload: dict[str, Any] = {
        "schema": "torchcst-e2e-mnist-conv-v1",
        "comparison_contract": {
            "oracle": "cst CONV-GROW-4",
            "same": [
                "MNIST split",
                "8->16 target conv topology",
                "batch tape family",
                "K=4->36 ladder",
                "birth steps",
                "AdamW learning rates",
            ],
            "changed": "per-offset rank-one blocks -> continuous Gaussian CSTLinear",
            "interpretation": "protocol reproduction; not same-family numerical reproduction",
        },
        "protocol": {
            "steps": protocol.steps,
            "batch_size": protocol.batch_size,
            "train_n": protocol.train_n,
            "eval_n": protocol.eval_n,
            "birth_steps": protocol.birth_steps,
            "birth_counts": birth_counts(protocol.birth_steps),
            "matched_jump_step": matched_jump_step(protocol.steps, protocol.birth_steps),
            "sigma_in": SIGMA_IN,
            "sigma_out": SIGMA_OUT,
            "candidate_pool": CANDIDATE_POOL,
        },
        "runs": [],
    }
    for arm in protocol.arms:
        for seed in seeds:
            payload["runs"].append(run_arm(arm, seed, protocol, data, device))
            atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    protocol = SMOKE_PROTOCOL if args.smoke else FULL_PROTOCOL
    seeds = tuple(args.seed) if args.seed else ((0,) if args.smoke else SEEDS)
    result = run(protocol, seeds, args.data_dir, args.output, torch.device(args.device))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
