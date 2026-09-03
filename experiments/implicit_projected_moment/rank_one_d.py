"""Test a rank-one square-root D-side signal on a tiny Gaussian CST map.

The difficult elementwise candidate is

``D_diag = Diag(abs(vec(grad_W)) + eps)``.

This experiment compares it with the tractable full-gradient outer-product
candidate

``D_rank = sqrt(g g.T) + eps I = g g.T / ||g|| + eps I``

for ``g = vec(grad_W)``.  The rank-one candidate is special because every
second-order contraction can be reconstructed from the dense-free P2
statistics

``b = J.T g`` and ``C = g contract H``

plus ``||g||``.  Dense tensors are used here only as a tiny correctness oracle.
The experiment measures both the algebraic recovery error and how different
the rank-one force is from the elementwise diagonal force.

Run from the repository root with, for example::

    PYTHONPATH=src python -m \
        experiments.implicit_projected_moment.rank_one_d --preset smoke
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.optim.p2 import contracted_p2_model


@dataclass(frozen=True)
class ExperimentConfig:
    """One deterministic rank-one D-side problem."""

    atoms: int = 2
    amplitude: float = 1.0e-2
    residual_scale: float = 1.0e-1
    spread: float = 0.25
    batch_size: int = 8
    n_in: int = 6
    n_out: int = 5
    sigma: float = 0.24
    radius: float = 1.0e-2
    epsilon: float = 1.0e-8
    directions: int = 16
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.atoms <= 0:
            raise ValueError("atoms must be positive")
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.spread < 0:
            raise ValueError("spread must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_in <= 1 or self.n_out <= 1:
            raise ValueError("n_in and n_out must exceed one")
        if self.sigma <= 0 or self.radius <= 0:
            raise ValueError("sigma and radius must be positive")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.directions <= 0:
            raise ValueError("directions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True)
class TrialResult:
    """Mechanism and approximation diagnostics for one problem."""

    config: ExperimentConfig
    parameter_dimension: int
    ambient_dimension: int
    gradient_norm: float
    p2_linear_relative_error: float
    p2_curvature_relative_error: float
    factored_gradient_norm_relative_error: float
    rank_sqrt_identity_relative_error: float
    jtdj_relative_error: float
    jtdh_relative_error: float
    htdh_relative_error: float
    maximum_compact_energy_relative_error: float
    maximum_compact_force_relative_error: float
    median_rank_vs_diag_force_cosine: float
    median_rank_vs_diag_force_relative_error: float
    median_rank_to_diag_energy_ratio: float
    diag_pullback_cross_atom_fraction: float
    rank_pullback_cross_atom_fraction: float
    rank_pullback_effective_rank: int


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _number(value: Tensor | float) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _relative_error(candidate: Tensor, reference: Tensor) -> Tensor:
    floor = torch.finfo(reference.dtype).tiny
    return torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(floor)


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
    return values / torch.linalg.vector_norm(values, dim=0, keepdim=True).clamp_min(
        torch.finfo(values.dtype).tiny
    )


def _initial_theta(config: ExperimentConfig, *, device: torch.device) -> Tensor:
    dtype = _torch_dtype(config.dtype)
    if config.atoms == 1:
        offsets = torch.zeros(1, device=device, dtype=dtype)
    else:
        offsets = torch.linspace(
            -0.5, 0.5, config.atoms, device=device, dtype=dtype
        ) * config.spread
    amplitudes = torch.full(
        (config.atoms,), config.amplitude, device=device, dtype=dtype
    )
    return torch.stack((amplitudes, 0.43 + offsets, 0.57 - offsets), dim=1)


def _block_diagonal(matrix: Tensor, atoms: int, fields: int = 3) -> Tensor:
    blocks = matrix.reshape(atoms, fields, atoms, fields)
    result = torch.zeros_like(blocks)
    indices = torch.arange(atoms, device=matrix.device)
    result[indices, :, indices, :] = blocks[indices, :, indices, :]
    return result.reshape_as(matrix)


def _cross_atom_fraction(matrix: Tensor, atoms: int) -> Tensor:
    floor = torch.finfo(matrix.dtype).tiny
    return torch.linalg.vector_norm(matrix - _block_diagonal(matrix, atoms)) / (
        torch.linalg.vector_norm(matrix).clamp_min(floor)
    )


def _full_block_matrix(blocks: Tensor) -> Tensor:
    atoms, fields, _ = blocks.shape
    result = blocks.new_zeros((atoms, fields, atoms, fields))
    indices = torch.arange(atoms, device=blocks.device)
    result[indices, :, indices, :] = blocks
    return result.reshape(atoms * fields, atoms * fields)


def _problem(config: ExperimentConfig):
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    theta_rows = _initial_theta(config, device=device)
    theta = theta_rows.flatten()
    mu_in = torch.linspace(0.0, 1.0, config.n_in, device=device, dtype=dtype)[:, None]
    mu_out = torch.linspace(0.0, 1.0, config.n_out, device=device, dtype=dtype)[:, None]
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

    weight = weight_fn(theta).detach()
    with torch.no_grad():
        outputs = F.linear(inputs, weight)
        residual = torch.randn(
            outputs.shape, generator=generator, device=device, dtype=dtype
        )
        residual = residual / residual.square().mean().sqrt()
        targets = outputs - config.residual_scale * residual

    weight_leaf = weight.requires_grad_(True)
    outputs = F.linear(inputs, weight_leaf)
    loss = F.mse_loss(outputs, targets)
    grad_output, grad_weight = torch.autograd.grad(loss, (outputs, weight_leaf))
    return (
        theta,
        theta_rows,
        inputs,
        grad_output.detach(),
        grad_weight.detach(),
        mu_in,
        mu_out,
        weight_fn,
        generator,
    )


def _candidate_directions(
    config: ExperimentConfig,
    *,
    reference: Tensor,
    generator: torch.Generator,
) -> list[Tensor]:
    scales = reference.new_tensor([1.0, config.sigma, config.sigma]).repeat(
        config.atoms
    )
    directions = []
    for _ in range(config.directions):
        raw = torch.randn(
            reference.shape,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        raw = raw / torch.linalg.vector_norm(raw).clamp_min(
            torch.finfo(raw.dtype).tiny
        )
        directions.append(config.radius * scales * raw)
    return directions


def run_trial(config: ExperimentConfig) -> TrialResult:
    """Run one dense oracle and compact rank-one recovery trial."""

    (
        theta,
        theta_rows,
        inputs,
        grad_output,
        grad_weight,
        mu_in,
        mu_out,
        weight_fn,
        generator,
    ) = _problem(config)
    flat_weight_fn = lambda point: weight_fn(point).flatten()
    jacobian = torch.func.jacrev(flat_weight_fn)(theta)
    hessian = torch.func.jacfwd(torch.func.jacrev(flat_weight_fn))(theta)
    ambient_gradient = grad_weight.flatten()
    gradient_norm = torch.linalg.vector_norm(ambient_gradient)
    if not bool(gradient_norm > 0):
        raise RuntimeError("the generated problem has a zero ambient gradient")

    linear = jacobian.transpose(0, 1) @ ambient_gradient
    curvature = torch.einsum("n,nij->ij", ambient_gradient, hessian)
    curvature = 0.5 * (curvature + curvature.T)

    def atom_score(atom: Tensor) -> Tensor:
        k_in = _columns(mu_in, atom[1:2][None, :], config.sigma)[:, 0]
        k_out = _columns(mu_out, atom[2:3][None, :], config.sigma)[:, 0]
        activation = inputs @ k_in
        output_projection = grad_output @ k_out
        return atom[0] * (activation * output_projection).sum()

    p2_model = contracted_p2_model(atom_score, theta_rows)
    p2_linear = p2_model.linear.flatten()
    p2_curvature = _full_block_matrix(p2_model.curvature)

    input_batch_gram = inputs @ inputs.T
    output_batch_gram = grad_output @ grad_output.T
    factored_gradient_norm = (
        input_batch_gram * output_batch_gram
    ).sum().clamp_min(0).sqrt()

    ambient_dimension = ambient_gradient.numel()
    identity = torch.eye(
        ambient_dimension, device=theta.device, dtype=theta.dtype
    )
    outer = ambient_gradient[:, None] * ambient_gradient[None, :]
    rank_sqrt = outer / gradient_norm
    rank_sqrt_error = _relative_error(rank_sqrt @ rank_sqrt, outer)

    dense_jtdj = jacobian.T @ rank_sqrt @ jacobian
    dense_jtdh = torch.einsum(
        "ni,nm,mab->iab", jacobian, rank_sqrt, hessian
    )
    dense_htdh = torch.einsum(
        "nij,nm,mab->ijab", hessian, rank_sqrt, hessian
    )
    compact_jtdj = linear[:, None] * linear[None, :] / gradient_norm
    compact_jtdh = torch.einsum(
        "i,ab->iab", linear, curvature
    ) / gradient_norm
    compact_htdh = torch.einsum(
        "ij,ab->ijab", curvature, curvature
    ) / gradient_norm

    diag_abs = torch.diag(ambient_gradient.abs() + config.epsilon)
    rank_with_epsilon = rank_sqrt + config.epsilon * identity
    diag_pullback = jacobian.T @ diag_abs @ jacobian
    rank_pullback = jacobian.T @ rank_with_epsilon @ jacobian
    pure_rank_eigenvalues = torch.linalg.eigvalsh(dense_jtdj)
    rank_tolerance = (
        256.0
        * torch.finfo(theta.dtype).eps
        * pure_rank_eigenvalues.abs().amax().clamp_min(1.0)
    )
    effective_rank = int((pure_rank_eigenvalues > rank_tolerance).sum())

    compact_energy_errors = []
    compact_force_errors = []
    force_cosines = []
    force_relative_errors = []
    energy_ratios = []
    for direction in _candidate_directions(
        config, reference=theta, generator=generator
    ):
        hd = torch.einsum("nij,i->nj", hessian, direction)
        delta = jacobian @ direction + 0.5 * hd @ direction
        visible = jacobian + hd

        score = linear @ direction + 0.5 * direction @ curvature @ direction
        pullback = linear + curvature @ direction
        dense_energy = 0.5 * delta @ rank_with_epsilon @ delta
        compact_energy = (
            0.5 * score.square() / gradient_norm
            + 0.5 * config.epsilon * delta.square().sum()
        )
        dense_force = visible.T @ rank_with_epsilon @ delta
        compact_force = (
            (score / gradient_norm) * pullback
            + config.epsilon * (visible.T @ delta)
        )
        diag_energy = 0.5 * delta @ diag_abs @ delta
        diag_force = visible.T @ diag_abs @ delta

        compact_energy_errors.append(_relative_error(compact_energy, dense_energy))
        compact_force_errors.append(_relative_error(compact_force, dense_force))
        force_cosines.append(_cosine(dense_force, diag_force))
        force_relative_errors.append(_relative_error(dense_force, diag_force))
        energy_ratios.append(
            dense_energy / diag_energy.clamp_min(torch.finfo(theta.dtype).tiny)
        )

    return TrialResult(
        config=config,
        parameter_dimension=theta.numel(),
        ambient_dimension=ambient_dimension,
        gradient_norm=_number(gradient_norm),
        p2_linear_relative_error=_number(_relative_error(p2_linear, linear)),
        p2_curvature_relative_error=_number(
            _relative_error(p2_curvature, curvature)
        ),
        factored_gradient_norm_relative_error=_number(
            _relative_error(factored_gradient_norm, gradient_norm)
        ),
        rank_sqrt_identity_relative_error=_number(rank_sqrt_error),
        jtdj_relative_error=_number(_relative_error(compact_jtdj, dense_jtdj)),
        jtdh_relative_error=_number(_relative_error(compact_jtdh, dense_jtdh)),
        htdh_relative_error=_number(_relative_error(compact_htdh, dense_htdh)),
        maximum_compact_energy_relative_error=max(
            _number(value) for value in compact_energy_errors
        ),
        maximum_compact_force_relative_error=max(
            _number(value) for value in compact_force_errors
        ),
        median_rank_vs_diag_force_cosine=statistics.median(
            _number(value) for value in force_cosines
        ),
        median_rank_vs_diag_force_relative_error=statistics.median(
            _number(value) for value in force_relative_errors
        ),
        median_rank_to_diag_energy_ratio=statistics.median(
            _number(value) for value in energy_ratios
        ),
        diag_pullback_cross_atom_fraction=_number(
            _cross_atom_fraction(diag_pullback, config.atoms)
        ),
        rank_pullback_cross_atom_fraction=_number(
            _cross_atom_fraction(rank_pullback, config.atoms)
        ),
        rank_pullback_effective_rank=effective_rank,
    )


def _finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _median(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float("nan") if not finite else statistics.median(finite)


def _comparison_summary(results: list[TrialResult]) -> dict[str, float]:
    return {
        "minimum_force_cosine": min(
            result.median_rank_vs_diag_force_cosine for result in results
        ),
        "median_force_cosine": _median(
            result.median_rank_vs_diag_force_cosine for result in results
        ),
        "median_force_relative_error": _median(
            result.median_rank_vs_diag_force_relative_error for result in results
        ),
        "median_energy_ratio": _median(
            result.median_rank_to_diag_energy_ratio for result in results
        ),
        "median_diag_pullback_cross_atom_fraction": _median(
            result.diag_pullback_cross_atom_fraction for result in results
        ),
        "median_rank_pullback_cross_atom_fraction": _median(
            result.rank_pullback_cross_atom_fraction for result in results
        ),
    }


def summarize(results: list[TrialResult]) -> dict[str, object]:
    """Aggregate algebraic checks and rank-one/diagonal differences."""

    if not results:
        raise ValueError("cannot summarize an empty result list")
    return {
        "trials": len(results),
        "mechanism_checks": {
            "max_p2_linear_relative_error": max(
                result.p2_linear_relative_error for result in results
            ),
            "max_p2_curvature_relative_error": max(
                result.p2_curvature_relative_error for result in results
            ),
            "max_factored_gradient_norm_relative_error": max(
                result.factored_gradient_norm_relative_error for result in results
            ),
            "max_rank_sqrt_identity_relative_error": max(
                result.rank_sqrt_identity_relative_error for result in results
            ),
            "max_jtdj_relative_error": max(
                result.jtdj_relative_error for result in results
            ),
            "max_jtdh_relative_error": max(
                result.jtdh_relative_error for result in results
            ),
            "max_htdh_relative_error": max(
                result.htdh_relative_error for result in results
            ),
            "max_compact_energy_relative_error": max(
                result.maximum_compact_energy_relative_error for result in results
            ),
            "max_compact_force_relative_error": max(
                result.maximum_compact_force_relative_error for result in results
            ),
            "max_rank_pullback_effective_rank": max(
                result.rank_pullback_effective_rank for result in results
            ),
        },
        "rank_one_vs_elementwise_diagonal": _comparison_summary(results),
        "by_atoms": {
            str(atoms): _comparison_summary(
                [result for result in results if result.config.atoms == atoms]
            )
            for atoms in sorted({result.config.atoms for result in results})
        },
        "by_amplitude": {
            f"{amplitude:.12g}": _comparison_summary(
                [
                    result
                    for result in results
                    if result.config.amplitude == amplitude
                ]
            )
            for amplitude in sorted(
                {result.config.amplitude for result in results}
            )
        },
    }


def _preset(name: str) -> dict[str, list[float | int]]:
    if name == "smoke":
        return {
            "atoms": [1, 2],
            "amplitudes": [1.0e-3, 1.0e-1],
            "residuals": [1.0e-3, 1.0e-1],
            "spreads": [0.0, 0.3],
            "batch_sizes": [8],
            "seeds": [0, 1],
        }
    if name == "pilot":
        return {
            "atoms": [1, 2, 4],
            "amplitudes": [1.0e-4, 1.0e-2, 1.0],
            "residuals": [1.0e-4, 1.0e-2, 1.0],
            "spreads": [0.0, 0.2, 0.6],
            "batch_sizes": [8, 32],
            "seeds": [0, 1, 2, 3],
        }
    raise ValueError(f"unknown preset {name!r}")


def run_sweep(
    *, preset: str, device: str, dtype: str, progress_every: int = 10
) -> list[TrialResult]:
    grid = _preset(preset)
    configs = [
        ExperimentConfig(
            atoms=int(atoms),
            amplitude=float(amplitude),
            residual_scale=float(residual),
            spread=float(spread),
            batch_size=int(batch_size),
            seed=int(seed),
            device=device,
            dtype=dtype,
        )
        for atoms in grid["atoms"]
        for amplitude in grid["amplitudes"]
        for residual in grid["residuals"]
        for spread in grid["spreads"]
        for batch_size in grid["batch_sizes"]
        for seed in grid["seeds"]
    ]
    results = []
    started = time.perf_counter()
    for index, config in enumerate(configs, start=1):
        results.append(run_trial(config))
        if progress_every > 0 and (index % progress_every == 0 or index == len(configs)):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(configs)} trials, {elapsed:.1f}s", flush=True)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.perf_counter()
    results = run_sweep(
        preset=args.preset,
        device=args.device,
        dtype=args.dtype,
        progress_every=args.progress_every,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "torchcst-implicit-rank-one-d-v1",
        "preset": args.preset,
        "device": args.device,
        "dtype": args.dtype,
        "gpu_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2, allow_nan=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {len(results)} trials to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
