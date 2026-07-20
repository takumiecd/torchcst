#!/usr/bin/env python3
"""First end-to-end MNIST exercise for the CSTF structural lifecycle.

This is a framework shakedown, not a research comparison.  Structural work is
triggered exclusively by policies from :mod:`torchcst.policy.catalog`; the training
loss is used only for ordinary backpropagation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torchcst.audit import Accounting, EconomyAudit
from torchcst.compute import CSTLinear, EntryLinear, RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.lab import BatchTape, Ledger, RngStreams
from torchcst.optim import parameter_groups
from torchcst.policy import LC, LC_anti, cRigL, cSET
from torchcst.policy.contract import Clock, Policy
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


IN_FEATURES = 28 * 28
HIDDEN_FEATURES = 64
CLASSES = 10
ENTRY_TARGET_K = 8_000
RANK_ONE_TARGET_K = 9
# 継続family (Gaussian kernel) は入出力gridそのものがneuron数なので、entryと
# 同じ幅を共有する: 784入力neuron, 64出力neuron.
CST_TARGET_K = 2_000
CST_INITIAL_K = 100
CST_SIGMA = 0.05
CST_INIT_WEIGHT_STD = 0.05
EVENT_INTERVAL = 50
FULL_UPDATES = 2_000
SMOKE_UPDATES = 200
# cst repo踏襲のamplitude/coordinate lr分離 (cst.optim.parameter_groups)。
AMPLITUDE_LR = 1.0e-3
COORDINATE_LR = 2.0e-4
DEFAULT_LR = 1.0e-3


@dataclass(frozen=True)
class ArmConfig:
    name: str
    family: str
    policy: str
    initial_k: int
    target_k: int
    event_interval: int = EVENT_INTERVAL
    birth_start_event: int = 1
    birth_end_event: int = 20
    birth_budget: int = 0
    freeze_event: int | None = 30
    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    drop_fraction_start: float = 0.3
    rewire_end_fraction: float = 0.75
    candidate_pool_size: int = 4_096
    # rank1-static診断 (TASK 2) 用: 静的rank-one atomはs/t座標を凍結し、振幅wと
    # 下流classifierだけを学習する固定random-feature基底として扱う。
    freeze_coordinates: bool = False


ARM_CONFIGS = (
    ArmConfig(
        "entry-static",
        "entry",
        "LC",
        ENTRY_TARGET_K,
        ENTRY_TARGET_K,
        birth_end_event=1,
        freeze_event=2,
        rent_ratio=0.0,
    ),
    ArmConfig(
        "entry-LC",
        "entry",
        "LC",
        400,
        ENTRY_TARGET_K,
        birth_budget=(ENTRY_TARGET_K - 400) // 20,
    ),
    ArmConfig("entry-cSET", "entry", "cSET", ENTRY_TARGET_K, ENTRY_TARGET_K),
    ArmConfig("entry-cRigL", "entry", "cRigL", ENTRY_TARGET_K, ENTRY_TARGET_K),
    # With integer atom budgets, twenty rank-one birth events cannot target nine
    # atoms.  These two arms therefore use the first eight events explicitly.
    ArmConfig(
        "rank1-LC",
        "rank1",
        "LC",
        1,
        RANK_ONE_TARGET_K,
        birth_end_event=8,
        birth_budget=1,
        freeze_event=18,
    ),
    ArmConfig(
        "rank1-LC-anti",
        "rank1",
        "LC_anti",
        1,
        RANK_ONE_TARGET_K,
        birth_end_event=8,
        birth_budget=1,
        freeze_event=18,
    ),
    ArmConfig(
        "rank1-static",
        "rank1",
        "LC",
        RANK_ONE_TARGET_K,
        RANK_ONE_TARGET_K,
        birth_end_event=1,
        freeze_event=2,
        rent_ratio=0.0,
        freeze_coordinates=True,
    ),
    # Continuous (Gaussian kernel) family.  Static pre-seeds every atom up
    # front with no structural churn, mirroring entry-static.  LC starts from
    # a small seed and grows the birth window toward the same target K, then
    # freezes/sweeps to quiescence exactly like entry-LC.
    ArmConfig(
        "cst-static",
        "cst",
        "LC",
        CST_TARGET_K,
        CST_TARGET_K,
        birth_end_event=1,
        freeze_event=2,
        rent_ratio=0.0,
    ),
    ArmConfig(
        "cst-LC",
        "cst",
        "LC",
        CST_INITIAL_K,
        CST_TARGET_K,
        birth_budget=(CST_TARGET_K - CST_INITIAL_K) // 20,
    ),
)
CONFIG_BY_NAME = {config.name: config for config in ARM_CONFIGS}


class MnistModel(nn.Module):
    def __init__(
        self,
        cst: EntryLinear | RankOneLinear | CSTLinear,
        classifier: nn.Linear,
    ) -> None:
        super().__init__()
        self.cst = cst
        self.activation = nn.ReLU()
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.activation(self.cst(x)))


def _read_bytes(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    compressed = path.with_name(path.name + ".gz")
    if compressed.exists():
        with gzip.open(compressed, "rb") as handle:
            return handle.read()
    raise FileNotFoundError(f"MNIST file not found: {path} or {compressed}")


def _parse_images(path: Path, limit: int) -> Tensor:
    raw = _read_bytes(path)
    if len(raw) < 16:
        raise ValueError(f"truncated IDX image file: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or rows * columns != IN_FEATURES:
        raise ValueError(f"unexpected IDX image header in {path}")
    expected = 16 + count * rows * columns
    if len(raw) != expected or count < limit:
        raise ValueError(f"invalid IDX image payload in {path}")
    values = torch.frombuffer(bytearray(raw), dtype=torch.uint8, offset=16)
    return values[: limit * rows * columns].clone().reshape(limit, -1).float().div_(255.0)


def _parse_labels(path: Path, limit: int) -> Tensor:
    raw = _read_bytes(path)
    if len(raw) < 8:
        raise ValueError(f"truncated IDX label file: {path}")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049 or len(raw) != 8 + count or count < limit:
        raise ValueError(f"invalid IDX label payload in {path}")
    values = torch.frombuffer(bytearray(raw), dtype=torch.uint8, offset=8)
    return values[:limit].clone().long()


def load_mnist(data_dir: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    train_x = _parse_images(data_dir / "train-images-idx3-ubyte", 10_000)
    train_y = _parse_labels(data_dir / "train-labels-idx1-ubyte", 10_000)
    test_x = _parse_images(data_dir / "t10k-images-idx3-ubyte", 2_000)
    test_y = _parse_labels(data_dir / "t10k-labels-idx1-ubyte", 2_000)
    return train_x, train_y, test_x, test_y


def _arm_root_seed(seed: int, arm: str) -> int:
    payload = f"torchcst-e2e-mnist-v1\0{seed}\0{arm}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _initial_birth(
    store: SynapseStore,
    config: ArmConfig,
    generator: torch.Generator,
) -> SynapseBirth:
    count = config.initial_k
    if config.family == "entry":
        lineages = torch.randperm(
            IN_FEATURES * HIDDEN_FEATURES, generator=generator
        )[:count]
        source = torch.div(lineages, HIDDEN_FEATURES, rounding_mode="floor")[:, None]
        target = torch.remainder(lineages, HIDDEN_FEATURES)[:, None]
        fan_in = count / HIDDEN_FEATURES
        scale = math.sqrt(2.0 / fan_in)
        weights = torch.randn(count, generator=generator).mul_(scale)
        return SynapseBirth(store.site, source, target, weights, lineages.to(torch.int64))
    if config.family == "cst":
        # Box座標は一様sample、振幅は小さいrandom (N(0, CST_INIT_WEIGHT_STD^2)) で
        # 種付けする。continuous familyの2000 atomは冗長性が高く、rank-oneの
        # ような少数atom collapseの心配は薄いので座標は通常通り学習させる。
        source = store.spec.domain_in.sample(count, generator)
        target = store.spec.domain_out.sample(count, generator)
        lineages = torch.arange(count, dtype=torch.int64)
        weights = torch.randn(count, generator=generator).mul_(CST_INIT_WEIGHT_STD)
        return SynapseBirth(store.site, source, target, weights, lineages)
    source = store.spec.domain_in.sample(count, generator)
    target = store.spec.domain_out.sample(count, generator)
    lineages = torch.arange(count, dtype=torch.int64)
    scale = math.sqrt(2.0 * HIDDEN_FEATURES / count)
    weights = torch.randn(count, generator=generator).mul_(scale)
    return SynapseBirth(store.site, source, target, weights, lineages)


def _cosine_drop(freeze_event: int) -> Callable[[Clock], float]:
    last_rewire = max(1, freeze_event - 1)

    def drop(clock: Clock) -> float:
        progress = (clock.event_index - 1) / max(1, last_rewire - 1)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * 0.3 * (1.0 + math.cos(math.pi * progress))

    return drop


def make_policy(config: ArmConfig, updates: int) -> tuple[Policy, int]:
    total_events = updates // config.event_interval
    if config.policy in {"cSET", "cRigL"}:
        freeze_event = max(2, math.ceil(total_events * config.rewire_end_fraction))
        common = {
            "event_interval": config.event_interval,
            "drop_fraction": _cosine_drop(freeze_event),
            "freeze_event": freeze_event,
            "initial_weight": 0.0,
        }
        catalog = (
            cSET(
                **common,
                bounds_in=IN_FEATURES,
                bounds_out=HIDDEN_FEATURES,
            )
            if config.policy == "cSET"
            else cRigL(
                **common,
                observe_window=1,
                pool_size=config.candidate_pool_size,
            )
        )
        return catalog.as_policy(), freeze_event

    common_lc = {
        "event_interval": config.event_interval,
        "birth_start_event": config.birth_start_event,
        "birth_end_event": config.birth_end_event,
        "birth_budget": config.birth_budget,
        "freeze_event": config.freeze_event,
        "immunity_events": config.immunity_events,
        "rent_ratio": config.rent_ratio,
        "strikes": config.strikes,
    }
    catalog = (
        LC_anti(**common_lc, observe_window=1, certificate_rank=1)
        if config.policy == "LC_anti"
        else LC(
            **common_lc,
            bounds_in=IN_FEATURES if config.family == "entry" else None,
            bounds_out=HIDDEN_FEATURES if config.family == "entry" else None,
        )
    )
    assert config.freeze_event is not None
    return catalog.as_policy(), config.freeze_event


def _grid_mu(n_max: int, device: torch.device) -> torch.Tensor:
    """[0,1]上の等間隔gridを (n_max, 1) のfloating bufferとして返す。"""
    return torch.linspace(0.0, 1.0, n_max, device=device).unsqueeze(1)


def build_arm(
    config: ArmConfig,
    seed: int,
    updates: int,
    device: torch.device,
) -> tuple[MnistModel, SynapseStore, StructuralEngine, torch.optim.Optimizer, dict[str, int]]:
    streams = RngStreams(_arm_root_seed(seed, config.name))
    if config.family == "entry":
        spec = RepresentationSpec.entry(bounds_in=IN_FEATURES, bounds_out=HIDDEN_FEATURES)
        dimensions = (1, 1)
    elif config.family == "cst":
        spec = RepresentationSpec.continuous(1, 1)
        dimensions = (1, 1)
    else:
        spec = RepresentationSpec.rank_one(IN_FEATURES, HIDDEN_FEATURES)
        dimensions = (IN_FEATURES, HIDDEN_FEATURES)
    store = SynapseStore(
        "mnist-cst",
        *dimensions,
        capacity=config.initial_k,
        spec=spec,
        device=device,
    )
    store.apply([_initial_birth(store, config, streams.get("init"))])
    if config.freeze_coordinates:
        # rank1-static診断 (TASK 2) の対策: 初期random方向の多様性が学習初期の
        # noisy勾配で崩壊するのを防ぐため、振幅wだけを学習するfixed random
        # feature基底として扱う。
        for coordinate in (store.s, store.t):
            if isinstance(coordinate, nn.Parameter):
                coordinate.requires_grad_(False)

    stores: dict[str, SynapseStore | NeuronStore] = {store.site: store}
    if config.family == "cst":
        in_neurons = NeuronStore(
            "mnist-cst-in",
            IN_FEATURES,
            mu=_grid_mu(IN_FEATURES, device),
            initial_live=IN_FEATURES,
            device=device,
        )
        out_neurons = NeuronStore(
            "mnist-cst-out",
            HIDDEN_FEATURES,
            mu=_grid_mu(HIDDEN_FEATURES, device),
            initial_live=HIDDEN_FEATURES,
            device=device,
        )
        kernel = GaussianKernel(CST_SIGMA, learnable=True)
        cst = CSTLinear(in_neurons, out_neurons, store, kernel)
        stores[in_neurons.site] = in_neurons
        stores[out_neurons.site] = out_neurons
    elif config.family == "entry":
        cst = EntryLinear(store, IN_FEATURES, HIDDEN_FEATURES)
    else:
        cst = RankOneLinear(store, IN_FEATURES, HIDDEN_FEATURES)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.seed("init"))
        classifier = nn.Linear(HIDDEN_FEATURES, CLASSES)
    model = MnistModel(cst, classifier).to(device)
    groups = parameter_groups(
        model,
        amplitude_lr=AMPLITUDE_LR,
        coordinate_lr=COORDINATE_LR,
        default_lr=DEFAULT_LR,
    )
    optimizer = torch.optim.Adam(groups)
    policy, freeze_event = make_policy(config, updates)
    # EconomyAuditのimmunity/strikesは、常に実際のretention courtが読み出す
    # 値と一致させる (config.immunity_eventsはRentCourt以外では無視される
    # arm設定値に過ぎない: 例えばMagnitudeCourtはimmunity_events=0固定)。
    retention = policy.retention
    audit = EconomyAudit(
        immunity_events=int(getattr(retention, "immunity_events", 0)),
        strikes=int(getattr(retention, "strikes", 1)),
    )
    engine = StructuralEngine(
        stores,
        policy,
        rng=streams.get("proposal"),
        modules={store.site: cst},
        optimizer=optimizer,
        audit_subscribers=[audit],
    )
    seeds = {
        "root": streams.root_seed,
        "init": streams.seed("init"),
        "proposal": streams.seed("proposal"),
        "effective_freeze_event": freeze_event,
    }
    engine._e2e_audit = audit
    return model, store, engine, optimizer, seeds


@torch.no_grad()
def accuracy(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, y.numel(), batch_size):
        batch_x = x[start : start + batch_size].to(device)
        batch_y = y[start : start + batch_size].to(device)
        correct += int((model(batch_x).argmax(1) == batch_y).sum())
    model.train()
    return correct / y.numel()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_arm(
    config: ArmConfig,
    seed: int,
    updates: int,
    tape: BatchTape,
    test_x: Tensor,
    test_y: Tensor,
    device: torch.device,
) -> dict[str, object]:
    model, store, engine, optimizer, arm_seeds = build_arm(
        config, seed, updates, device
    )
    audit: EconomyAudit = engine._e2e_audit
    initial_k = int(store.view().ids.numel())
    curve: list[dict[str, float | int]] = []
    _sync(device)
    started = time.perf_counter()
    for update, (batch_x, batch_y) in enumerate(tape, start=1):
        if update > updates:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        engine.begin_update()
        loss = F.cross_entropy(model(batch_x), batch_y)
        loss.backward()
        engine.observe_microbatch(1.0)
        engine.finalize_backward()
        optimizer.step()
        # Release the completed autograd graph before a structural event may
        # enlarge a Parameter in place.  A live old graph pins AccumulateGrad's
        # former shape even though the Parameter storage has grown.
        del loss
        # Materialize MPS writes before an event copies mass to CPU for policy
        # adjudication/audit.  Without this boundary, an immediate second
        # indexed read can observe an unfinished MPS command buffer.
        if update % config.event_interval == 0:
            _sync(device)
        engine.step()
        if update % 200 == 0 or update == updates:
            curve.append(
                {
                    "update": update,
                    "test_accuracy": accuracy(model, test_x, test_y, device),
                }
            )
    _sync(device)
    elapsed = time.perf_counter() - started

    final_accuracy = curve[-1]["test_accuracy"]
    assert isinstance(final_accuracy, float)
    k_events = list(audit.k_series(store.site))
    k_values = [initial_k, *k_events]
    active = Accounting.active_params(store)
    atom_ops = [op for _, op in engine.op_log()]
    birth_ops = [op for op in atom_ops if isinstance(op, SynapseBirth)]
    death_ops = [op for op in atom_ops if isinstance(op, SynapseDeath)]
    config_payload = asdict(config)
    config_payload["effective_freeze_event"] = arm_seeds.pop(
        "effective_freeze_event"
    )
    return {
        "accuracy_curve": curve,
        "active_params": active.total,
        "accounting": {
            "active_cst_params": active.total,
            "dense_head_params": sum(p.numel() for p in model.classifier.parameters()),
            "gamma": Accounting.gamma(store.spec, IN_FEATURES, HIDDEN_FEATURES),
            "optimizer_state_bytes": Accounting.optimizer_state_bytes(optimizer),
        },
        "arm": config.name,
        "churn": {
            "by_event": list(audit.churn),
            "total": sum(audit.churn),
        },
        "config": config_payload,
        "final_test_accuracy": final_accuracy,
        "k": {
            "by_event": k_events,
            "final": k_values[-1],
            "initial": initial_k,
            "maximum": max(k_values),
            "minimum": min(k_values),
        },
        "op_counts": {
            "birth_atoms": sum(int(op.w.numel()) for op in birth_ops),
            "birth_ops": len(birth_ops),
            "death_atoms": sum(int(op.ids.numel()) for op in death_ops),
            "death_ops": len(death_ops),
        },
        "quiescence_event": audit.quiescence_event,
        "runtime_seconds": elapsed,
        "seed": seed,
        "stream_seeds": arm_seeds,
        "tape_hash": tape.tape_hash,
        "thrash": {
            "count": audit.thrash_count,
            "immune_prunes": audit.immune_prunes,
            "maturity_events": audit.maturity_events,
            "prune_count": audit.prune_count,
            "rate": audit.thrash_rate,
        },
    }


def _split_values(values: Iterable[str]) -> list[str]:
    return [part for value in values for part in value.split(",") if part]


def _default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../cst/data/MNIST/raw"),
    )
    parser.add_argument("--seeds", nargs="+", default=["0"])
    parser.add_argument("--arms", nargs="+", default=[",".join(CONFIG_BY_NAME)])
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/e2e_mnist.json")
    )
    return parser.parse_args()


def _write_results(path: Path, payload: dict[str, object]) -> str:
    digest = hashlib.sha256(Ledger._canonical_bytes(payload)).hexdigest()
    document = {**payload, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Ledger._canonical_bytes(document))
    return digest


def _print_table(runs: list[dict[str, object]]) -> None:
    headers = (
        "arm/seed",
        "accuracy",
        "params",
        "K_final",
        "thrash",
        "immune_prunes",
        "quiet",
    )
    rows = []
    for run in runs:
        thrash = run["thrash"]
        k = run["k"]
        assert isinstance(thrash, dict) and isinstance(k, dict)
        rows.append(
            (
                f"{run['arm']}/{run['seed']}",
                f"{float(run['final_test_accuracy']):.4f}",
                str(run["active_params"]),
                str(k["final"]),
                f"{float(thrash['rate']):.3f}",
                str(thrash["immune_prunes"]),
                "-" if run["quiescence_event"] is None else str(run["quiescence_event"]),
            )
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in _split_values(args.seeds)]
    arm_names = _split_values(args.arms)
    unknown = sorted(set(arm_names).difference(CONFIG_BY_NAME))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}; choices={list(CONFIG_BY_NAME)}")
    device = torch.device(args.device)
    updates = SMOKE_UPDATES if args.smoke else FULL_UPDATES
    train_x, train_y, test_x, test_y = load_mnist(args.data_dir)
    batches_per_epoch = math.ceil(train_y.numel() / 128)
    tape_epochs = math.ceil(updates / batches_per_epoch)
    runs: list[dict[str, object]] = []
    tape_hashes: dict[int, str] = {}

    for seed in seeds:
        tape = BatchTape(
            train_x,
            train_y,
            batch_size=128,
            streams=RngStreams(seed),
            epochs=tape_epochs,
        )
        tape_hashes[seed] = tape.tape_hash
        for arm_name in arm_names:
            print(f"running {arm_name} seed={seed} updates={updates} device={device}")
            runs.append(
                run_arm(
                    CONFIG_BY_NAME[arm_name],
                    seed,
                    updates,
                    tape,
                    test_x,
                    test_y,
                    device,
                )
            )

    script = Path(__file__).resolve()
    script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "provenance": {
            "data_dir": str(args.data_dir),
            "script": "experiments/e2e_mnist.py",
            "script_sha256": script_sha256,
        },
        "runs": runs,
        "schema": "torchcst-e2e-mnist-v1",
        "settings": {
            "arms": arm_names,
            "batch_size": 128,
            "device": str(device),
            "seeds": seeds,
            "smoke": bool(args.smoke),
            "tape_hashes": {str(seed): value for seed, value in tape_hashes.items()},
            "test_examples": 2_000,
            "train_examples": 10_000,
            "updates": updates,
        },
    }
    digest = _write_results(args.output, payload)
    print(f"wrote {args.output} sha256={digest}")
    _print_table(runs)


if __name__ == "__main__":
    main()
