"""Measure projected EMA fidelity against a never-compressed dense oracle.

This experiment deliberately does not solve the quartic CST step problem.  A
shared, first-order trust-radius trajectory supplies the accepted steps.  The
only experimental variable is how the first- and D-side EMA states are carried
between the moving CST-visible spaces.

The dense reference retains ambient EMA vectors for the duration of the tiny
oracle problem.  The proposed state retains only coordinates and old-point
metadata.  Dense Jacobians and Hessians are materialized solely to evaluate the
oracle; a production implementation would replace them with composed
push/pull contractions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ExperimentConfig:
    """One deterministic moving-frame EMA problem."""

    atoms: int = 2
    beta1: float = 0.9
    beta2: float = 0.99
    radius: float = 1.0e-2
    steps: int = 32
    batch_size: int = 16
    n_in: int = 6
    n_out: int = 5
    sigma: float = 0.24
    epsilon: float = 1.0e-8
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.atoms <= 0:
            raise ValueError("atoms must be positive")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must lie in [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must lie in [0, 1)")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.steps < 2:
            raise ValueError("steps must be at least two")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_in <= 1 or self.n_out <= 1:
            raise ValueError("n_in and n_out must exceed one")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True)
class _CompressedState:
    """Persistent compact state; no ambient vector or visible matrix is kept."""

    alpha: Tensor
    gamma: Tensor
    old_theta: Tensor
    old_step: Tensor
    accepted_tangent: bool


@dataclass(frozen=True)
class TrialResult:
    """EMA fidelity diagnostics for one prescribed trajectory."""

    config: ExperimentConfig
    parameter_dimension: int
    ambient_dimension: int
    initial_loss: float
    final_loss: float
    accepted_first_median_relative_error: float
    accepted_first_maximum_relative_error: float
    accepted_second_median_relative_error: float
    accepted_second_maximum_relative_error: float
    base_first_median_relative_error: float
    base_first_maximum_relative_error: float
    base_second_median_relative_error: float
    base_second_maximum_relative_error: float
    accepted_first_minimum_cosine: float
    accepted_second_minimum_cosine: float
    base_first_minimum_cosine: float
    base_second_minimum_cosine: float
    maximum_recompression_identity_error: float


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _number(value: Tensor | float) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _relative_error(candidate: Tensor, reference: Tensor) -> Tensor:
    floor = torch.finfo(reference.dtype).tiny
    denominator = torch.linalg.vector_norm(reference).clamp_min(floor)
    return torch.linalg.vector_norm(candidate - reference) / denominator


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    floor = torch.finfo(left.dtype).tiny
    denominator = (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(floor)
    return (left.flatten() @ right.flatten()) / denominator


def _columns(query: Tensor, centers: Tensor, sigma: float) -> Tensor:
    squared = (query[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
    logits = -0.5 * squared / sigma**2
    logits = logits - logits.amax(dim=0, keepdim=True)
    values = logits.exp()
    floor = torch.finfo(values.dtype).tiny
    return values / torch.linalg.vector_norm(
        values, dim=0, keepdim=True
    ).clamp_min(floor)


def _initial_theta(config: ExperimentConfig, *, device: torch.device) -> Tensor:
    dtype = _torch_dtype(config.dtype)
    if config.atoms == 1:
        offsets = torch.zeros(1, device=device, dtype=dtype)
    else:
        offsets = torch.linspace(
            -0.15, 0.15, config.atoms, device=device, dtype=dtype
        )
    amplitudes = torch.full(
        (config.atoms,), 0.15, device=device, dtype=dtype
    )
    return torch.stack((amplitudes, 0.4 + offsets, 0.6 - offsets), dim=1).flatten()


def _target_theta(initial: Tensor, atoms: int) -> Tensor:
    target = initial.reshape(atoms, 3).clone()
    target[:, 0] *= 1.2
    target[:, 1] += torch.linspace(
        0.08, -0.06, atoms, device=target.device, dtype=target.dtype
    )
    target[:, 2] += torch.linspace(
        -0.07, 0.09, atoms, device=target.device, dtype=target.dtype
    )
    return target.flatten()


def _geometry(weight_fn, theta: Tensor) -> tuple[Tensor, Tensor]:
    parameter_dimension = theta.numel()
    jacobian = torch.func.jacrev(weight_fn)(theta).reshape(
        -1, parameter_dimension
    )
    hessian = torch.func.hessian(weight_fn)(theta).reshape(
        -1, parameter_dimension, parameter_dimension
    )
    return jacobian, hessian


def _accepted_frame(jacobian: Tensor, hessian: Tensor, step: Tensor) -> Tensor:
    return jacobian + torch.einsum("nij,i->nj", hessian, step)


def _frame_from_state(weight_fn, state: _CompressedState) -> Tensor:
    jacobian, hessian = _geometry(weight_fn, state.old_theta)
    if state.accepted_tangent:
        return _accepted_frame(jacobian, hessian, state.old_step)
    return jacobian


def _coordinates(frame: Tensor, ambient: Tensor) -> Tensor:
    gram = frame.transpose(0, 1) @ frame
    numerator = frame.transpose(0, 1) @ ambient
    return torch.linalg.pinv(gram, hermitian=True) @ numerator


def _rank_one_force(
    gradient: Tensor,
    displacement: Tensor,
    epsilon: float,
) -> Tensor:
    floor = torch.finfo(gradient.dtype).tiny
    gradient_norm = torch.linalg.vector_norm(gradient).clamp_min(floor)
    return gradient * ((gradient @ displacement) / gradient_norm) + (
        epsilon * displacement
    )


def _summary_pair(values: list[Tensor]) -> tuple[float, float]:
    numbers = [_number(value) for value in values]
    return statistics.median(numbers), max(numbers)


def run_trial(config: ExperimentConfig) -> TrialResult:
    """Run one shared trajectory and compare both compressed EMA carriers."""

    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    theta = _initial_theta(config, device=device)
    target_theta = _target_theta(theta, config.atoms)
    mu_in = torch.linspace(
        0.0, 1.0, config.n_in, device=device, dtype=dtype
    )[:, None]
    mu_out = torch.linspace(
        0.0, 1.0, config.n_out, device=device, dtype=dtype
    )[:, None]
    inputs = torch.randn(
        config.batch_size,
        config.n_in,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    def weight_fn(point: Tensor) -> Tensor:
        local = point.reshape(config.atoms, 3)
        k_in = _columns(mu_in, local[:, 1:2], config.sigma)
        k_out = _columns(mu_out, local[:, 2:3], config.sigma)
        return (k_out * local[:, 0]) @ k_in.transpose(0, 1)

    targets = F.linear(inputs, weight_fn(target_theta)).detach()
    ambient_dimension = config.n_in * config.n_out
    parameter_dimension = theta.numel()
    dense_first = torch.zeros(ambient_dimension, device=device, dtype=dtype)
    dense_second = torch.zeros_like(dense_first)
    states: dict[str, _CompressedState | None] = {
        "accepted": None,
        "base": None,
    }
    errors = {
        "accepted": {"first": [], "second": []},
        "base": {"first": [], "second": []},
    }
    cosines = {
        "accepted": {"first": [], "second": []},
        "base": {"first": [], "second": []},
    }
    identity_errors: list[Tensor] = []
    coordinate_scales = theta.new_tensor([1.0, config.sigma, config.sigma]).repeat(
        config.atoms
    )
    initial_loss = None
    final_loss = None

    for _ in range(config.steps):
        weight = weight_fn(theta).detach().requires_grad_(True)
        loss = F.mse_loss(F.linear(inputs, weight), targets)
        if initial_loss is None:
            initial_loss = _number(loss)
        final_loss = _number(loss)
        (weight_gradient,) = torch.autograd.grad(loss, weight)
        gradient = weight_gradient.detach().flatten()
        jacobian, hessian = _geometry(weight_fn, theta)

        scaled_gradient = coordinate_scales * (
            jacobian.transpose(0, 1) @ gradient
        )
        floor = torch.finfo(dtype).tiny
        dimensionless_step = -config.radius * scaled_gradient / (
            torch.linalg.vector_norm(scaled_gradient).clamp_min(floor)
        )
        step = coordinate_scales * dimensionless_step
        displacement = jacobian @ step + 0.5 * torch.einsum(
            "nij,i,j->n", hessian, step, step
        )
        accepted_frame = _accepted_frame(jacobian, hessian, step)
        current_force = _rank_one_force(
            gradient, displacement, config.epsilon
        )

        dense_first_candidate = config.beta1 * dense_first + (
            1.0 - config.beta1
        ) * gradient
        dense_second_candidate = config.beta2 * dense_second + (
            1.0 - config.beta2
        ) * current_force
        dense_first_pullback = (
            accepted_frame.transpose(0, 1) @ dense_first_candidate
        )
        dense_second_pullback = (
            accepted_frame.transpose(0, 1) @ dense_second_candidate
        )

        next_states: dict[str, _CompressedState] = {}
        for name, state in states.items():
            if state is None:
                compact_first = torch.zeros_like(dense_first)
                compact_second = torch.zeros_like(dense_second)
            else:
                old_frame = _frame_from_state(weight_fn, state)
                compact_first = old_frame @ state.alpha
                compact_second = old_frame @ state.gamma

            compact_first_candidate = config.beta1 * compact_first + (
                1.0 - config.beta1
            ) * gradient
            compact_second_candidate = config.beta2 * compact_second + (
                1.0 - config.beta2
            ) * current_force
            compact_first_pullback = (
                accepted_frame.transpose(0, 1) @ compact_first_candidate
            )
            compact_second_pullback = (
                accepted_frame.transpose(0, 1) @ compact_second_candidate
            )
            errors[name]["first"].append(
                _relative_error(compact_first_pullback, dense_first_pullback)
            )
            errors[name]["second"].append(
                _relative_error(compact_second_pullback, dense_second_pullback)
            )
            cosines[name]["first"].append(
                _cosine(compact_first_pullback, dense_first_pullback)
            )
            cosines[name]["second"].append(
                _cosine(compact_second_pullback, dense_second_pullback)
            )

            compression_frame = accepted_frame if name == "accepted" else jacobian
            alpha = _coordinates(compression_frame, compact_first_candidate)
            gamma = _coordinates(compression_frame, compact_second_candidate)
            first_numerator = (
                compression_frame.transpose(0, 1) @ compact_first_candidate
            )
            second_numerator = (
                compression_frame.transpose(0, 1) @ compact_second_candidate
            )
            identity_errors.extend(
                (
                    _relative_error(
                        compression_frame.transpose(0, 1)
                        @ (compression_frame @ alpha),
                        first_numerator,
                    ),
                    _relative_error(
                        compression_frame.transpose(0, 1)
                        @ (compression_frame @ gamma),
                        second_numerator,
                    ),
                )
            )
            next_states[name] = _CompressedState(
                alpha=alpha.detach(),
                gamma=gamma.detach(),
                old_theta=theta.detach(),
                old_step=step.detach(),
                accepted_tangent=name == "accepted",
            )

        states = next_states
        dense_first = dense_first_candidate.detach()
        dense_second = dense_second_candidate.detach()
        theta = (theta + step).detach()

    accepted_first = _summary_pair(errors["accepted"]["first"])
    accepted_second = _summary_pair(errors["accepted"]["second"])
    base_first = _summary_pair(errors["base"]["first"])
    base_second = _summary_pair(errors["base"]["second"])
    if initial_loss is None or final_loss is None:
        raise RuntimeError("EMA transport trial produced no observations")
    return TrialResult(
        config=config,
        parameter_dimension=parameter_dimension,
        ambient_dimension=ambient_dimension,
        initial_loss=initial_loss,
        final_loss=final_loss,
        accepted_first_median_relative_error=accepted_first[0],
        accepted_first_maximum_relative_error=accepted_first[1],
        accepted_second_median_relative_error=accepted_second[0],
        accepted_second_maximum_relative_error=accepted_second[1],
        base_first_median_relative_error=base_first[0],
        base_first_maximum_relative_error=base_first[1],
        base_second_median_relative_error=base_second[0],
        base_second_maximum_relative_error=base_second[1],
        accepted_first_minimum_cosine=min(
            _number(value) for value in cosines["accepted"]["first"]
        ),
        accepted_second_minimum_cosine=min(
            _number(value) for value in cosines["accepted"]["second"]
        ),
        base_first_minimum_cosine=min(
            _number(value) for value in cosines["base"]["first"]
        ),
        base_second_minimum_cosine=min(
            _number(value) for value in cosines["base"]["second"]
        ),
        maximum_recompression_identity_error=max(
            _number(value) for value in identity_errors
        ),
    )


def _aggregate(results: list[TrialResult], prefix: str) -> dict[str, float]:
    first_medians = [
        getattr(result, f"{prefix}_first_median_relative_error")
        for result in results
    ]
    second_medians = [
        getattr(result, f"{prefix}_second_median_relative_error")
        for result in results
    ]
    return {
        "median_first_numerator_relative_error": statistics.median(first_medians),
        "maximum_first_numerator_relative_error": max(
            getattr(result, f"{prefix}_first_maximum_relative_error")
            for result in results
        ),
        "median_second_numerator_relative_error": statistics.median(second_medians),
        "maximum_second_numerator_relative_error": max(
            getattr(result, f"{prefix}_second_maximum_relative_error")
            for result in results
        ),
        "minimum_first_numerator_cosine": min(
            getattr(result, f"{prefix}_first_minimum_cosine")
            for result in results
        ),
        "minimum_second_numerator_cosine": min(
            getattr(result, f"{prefix}_second_minimum_cosine")
            for result in results
        ),
    }


def summarize(results: list[TrialResult]) -> dict[str, object]:
    """Aggregate trial-level fidelity metrics."""

    if not results:
        raise ValueError("cannot summarize an empty result list")
    moving = [
        result
        for result in results
        if result.config.beta1 > 0.0 or result.config.beta2 > 0.0
    ]
    controls = [
        result
        for result in results
        if result.config.beta1 == 0.0 and result.config.beta2 == 0.0
    ]
    by_radius = {}
    for radius in sorted({result.config.radius for result in moving}):
        group = [result for result in moving if result.config.radius == radius]
        by_radius[str(radius)] = {
            "accepted_frame": _aggregate(group, "accepted"),
            "base_frame": _aggregate(group, "base"),
        }
    by_atoms = {}
    for atoms in sorted({result.config.atoms for result in moving}):
        group = [result for result in moving if result.config.atoms == atoms]
        by_atoms[str(atoms)] = {
            "accepted_frame": _aggregate(group, "accepted"),
            "base_frame": _aggregate(group, "base"),
        }
    return {
        "trials": len(results),
        "moving_ema_trials": len(moving),
        "zero_history_controls": len(controls),
        "mechanism_checks": {
            "maximum_recompression_identity_error": max(
                result.maximum_recompression_identity_error for result in results
            ),
            "maximum_zero_history_first_error": max(
                (
                    result.accepted_first_maximum_relative_error
                    for result in controls
                ),
                default=0.0,
            ),
            "maximum_zero_history_second_error": max(
                (
                    result.accepted_second_maximum_relative_error
                    for result in controls
                ),
                default=0.0,
            ),
        },
        "accepted_frame_vs_dense": _aggregate(moving, "accepted"),
        "base_frame_vs_dense": _aggregate(moving, "base"),
        "by_radius": by_radius,
        "by_atoms": by_atoms,
    }


def _configs(preset: str, args: argparse.Namespace) -> list[ExperimentConfig]:
    if preset == "single":
        return [
            ExperimentConfig(
                atoms=args.atoms,
                beta1=args.beta1,
                beta2=args.beta2,
                radius=args.radius,
                steps=args.steps,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
            )
        ]
    seeds = (0, 1) if preset == "smoke" else tuple(range(8))
    steps = 24 if preset == "smoke" else 64
    configurations = []
    for atoms in (1, 2):
        for radius in (1.0e-3, 1.0e-2, 5.0e-2):
            for beta1, beta2 in ((0.0, 0.0), (0.9, 0.99)):
                for seed in seeds:
                    configurations.append(
                        ExperimentConfig(
                            atoms=atoms,
                            beta1=beta1,
                            beta2=beta2,
                            radius=radius,
                            steps=steps,
                            seed=seed,
                            device=args.device,
                            dtype=args.dtype,
                        )
                    )
    return configurations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=("single", "smoke", "pilot"), default="smoke"
    )
    parser.add_argument("--atoms", type=int, default=2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--radius", type=float, default=1.0e-2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    configurations = _configs(args.preset, args)
    started = time.perf_counter()
    results = []
    for index, configuration in enumerate(configurations, start=1):
        results.append(run_trial(configuration))
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"completed {index}/{len(configurations)} trials", flush=True)
    payload = {
        "preset": args.preset,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize(results),
        "trials": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
