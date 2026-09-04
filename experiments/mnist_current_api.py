"""Reproduce the fixed-K MNIST experiment through the public torchcst API.

This runner deliberately uses the current Euclidean parameter-space trust
region.  It matches the successful experiment's normalized Gaussian kernels,
amplitude-dependent output width, small amplitude initialization, data split,
and compact-moment optimizer settings without importing the historical
experiment repository.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from torchcst import (
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    CSTOptimizer,
    FullQuartic,
    Gaussian,
    ImplicitAdamConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/mnist_current_api.json"


@dataclass(frozen=True)
class ExperimentConfig:
    atoms: int = 64
    steps: int = 128
    batch_size: int = 128
    train_size: int = 8192
    test_size: int = 2000
    input_sigma: float = 0.25
    output_sigma: float = 0.10
    sigma_explore: float = math.inf
    tau: float = 0.005
    temperature: float = 0.25
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    trust_radius: float = 0.25
    solver_starts: int = 4
    solver_max_iter: int = 80
    seed: int = 17


def read_idx(path: Path) -> Tensor:
    """Read an MNIST IDX tensor without NumPy or torchvision."""

    compressed = path.with_name(path.name + ".gz")
    if path.exists():
        blob = path.read_bytes()
    elif compressed.exists():
        blob = gzip.decompress(compressed.read_bytes())
    else:
        raise FileNotFoundError(path)
    if len(blob) < 8:
        raise ValueError(f"invalid IDX file: {path}")
    magic, count = struct.unpack(">II", blob[:8])
    if magic >> 8 != 0x000008:
        raise ValueError(f"unsupported IDX element type: {path}")
    dimensions = (magic & 0xFF) - 1
    if dimensions < 0:
        raise ValueError(f"invalid IDX dimensions: {path}")
    header_size = 8 + 4 * dimensions
    if len(blob) < header_size:
        raise ValueError(f"truncated IDX header: {path}")
    shape = struct.unpack(f">{dimensions}I", blob[8:header_size])
    flat = torch.frombuffer(bytearray(blob[header_size:]), dtype=torch.uint8)
    expected = count * math.prod(shape)
    if flat.numel() != expected:
        raise ValueError(f"IDX payload has the wrong size: {path}")
    return flat.reshape(count, *shape)


def load_mnist(
    root: Path,
    *,
    train_size: int,
    test_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load and standardize the fixed MNIST subsets used by the reference."""

    def prepare(prefix: str, limit: int) -> tuple[Tensor, Tensor]:
        images = read_idx(root / f"{prefix}-images-idx3-ubyte")
        labels = read_idx(root / f"{prefix}-labels-idx1-ubyte")
        if limit > images.shape[0] or limit > labels.shape[0]:
            raise ValueError(f"requested {limit} examples from {prefix}")
        inputs = images[:limit].to(dtype=torch.float32).reshape(limit, -1)
        inputs = (inputs / 255.0 - 0.1307) / 0.3081
        return inputs, labels[:limit].to(dtype=torch.int64)

    train_inputs, train_labels = prepare("train", train_size)
    test_inputs, test_labels = prepare("t10k", test_size)
    return train_inputs, train_labels, test_inputs, test_labels


def build_model(config: ExperimentConfig, device: torch.device) -> CSTLinear:
    """Build the four-coordinate fixed-tau model from the successful family."""

    torch.manual_seed(config.seed)
    model = CSTLinear(
        Chart.grid((28, 28)),
        Chart.linspace(10),
        atoms=config.atoms,
        kernel=AmplitudeBandwidthSeparable(
            input_profile=Gaussian(config.input_sigma),
            output_profile=Gaussian(config.output_sigma),
            sigma_explore=config.sigma_explore,
            tau=config.tau,
            temperature=config.temperature,
        ),
        atom_init="uniform",
        backend="factored",
        dtype=torch.float32,
    )
    return model.to(device)


def build_optimizer(model: CSTLinear, config: ExperimentConfig) -> CSTOptimizer:
    return CSTOptimizer(
        model,
        cst=ImplicitAdamConfig(
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            trust_radius=config.trust_radius,
            quartic=FullQuartic(
                starts=config.solver_starts,
                max_iter=config.solver_max_iter,
            ),
        ),
        dense=None,
    )


@torch.no_grad()
def evaluate(model: CSTLinear, inputs: Tensor, labels: Tensor) -> dict[str, float]:
    logits = model(inputs)
    return {
        "loss": float(F.cross_entropy(logits, labels)),
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_seed(
    config: ExperimentConfig,
    dataset: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    train_inputs, train_labels, test_inputs, test_labels = (
        value.to(device) for value in dataset
    )
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    checkpoints: list[dict[str, Any]] = [
        {"step": 0, **evaluate(model, test_inputs, test_labels)}
    ]
    trace: list[dict[str, Any]] = []
    permutation = torch.randperm(
        config.train_size,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    checkpoint_steps = {1, 4, 8, 16, 32, 64, config.steps}
    _synchronize(device)
    started = time.perf_counter()

    for step_index in range(config.steps):
        start = (step_index * config.batch_size) % config.train_size
        indices = permutation[start : start + config.batch_size].to(device)
        inputs = train_inputs.index_select(0, indices)
        labels = train_labels.index_select(0, indices)

        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()
        result = optimizer.last_step
        if result is None:
            raise RuntimeError("optimizer did not publish step diagnostics")
        solve = result.site_results[0]
        row = {
            "step": step_index + 1,
            "loss": float(loss.detach()),
            "solver_objective": float(solve.objective),
            "solver_evaluations": solve.evaluations,
            "solver_iterations": solve.iterations,
            "solver_start_index": solve.start_index,
            "solver_converged": solve.converged,
            "solver_projected_gradient_norm": float(
                solve.projected_gradient_norm
            ),
            "on_trust_boundary": solve.on_boundary,
        }
        trace.append(row)
        if step_index + 1 in checkpoint_steps:
            checkpoint = {
                "step": step_index + 1,
                **evaluate(model, test_inputs, test_labels),
            }
            checkpoints.append(checkpoint)
            print(
                f"seed={config.seed} step={step_index + 1:03d} "
                f"train_loss={row['loss']:.5f} "
                f"test_accuracy={100.0 * checkpoint['accuracy']:.2f}%",
                flush=True,
            )

    _synchronize(device)
    return {
        "config": asdict(config),
        "initial": checkpoints[0],
        "final": checkpoints[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoints": checkpoints,
        "trace": trace,
    }


def _default_data_root() -> Path:
    local = ROOT / "data/MNIST/raw"
    sibling = ROOT.parent / "cst/data/MNIST/raw"
    return local if local.exists() else sibling


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_default_data_root())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--trust-radius", type=float, default=0.25)
    parser.add_argument("--solver-starts", type=int, default=4)
    parser.add_argument("--solver-max-iter", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        atoms=args.atoms,
        steps=args.steps,
        batch_size=args.batch_size,
        train_size=args.train_size,
        test_size=args.test_size,
        trust_radius=args.trust_radius,
        solver_starts=args.solver_starts,
        solver_max_iter=args.solver_max_iter,
        seed=args.seed,
    )
    device = torch.device(args.device)
    dataset = load_mnist(
        args.data,
        train_size=config.train_size,
        test_size=config.test_size,
    )
    payload = {
        "protocol": {
            "public_api_only": True,
            "trust_region": "euclidean_parameter_space",
            "normalized_gaussian": True,
            "amplitude_init_std": "0.1/sqrt(K)",
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
        },
        "result": run_seed(config, dataset, device=device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("protocol", "environment")}, indent=2))
    print(json.dumps(payload["result"]["final"], indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
