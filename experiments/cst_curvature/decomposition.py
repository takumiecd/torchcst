"""Separate CST map curvature from ordinary loss curvature.

The experiment uses one Gaussian CST linear layer and a mean-squared-error
loss.  MSE is deliberately useful here: as a function of the materialized
weight ``W`` it is exactly quadratic.  Consequently the four reported loss
change predictions have an unambiguous interpretation:

``p1``
    First order in both ``theta -> W`` and ``W -> loss``.
``p2_cst``
    Second order in ``theta -> W``, still first order in ``W -> loss``.
``p_weight_exact``
    Exact ``theta -> W`` finite change, still first order in ``W -> loss``.
``p_actual``
    The actual finite loss change.

The experiment never changes the optimizer implementation.  It studies one
plain SGD direction at a time and keeps the research code outside ``src``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from itertools import product
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.representation import GaussianFactor


@dataclass(frozen=True)
class TrialConfig:
    """One deterministic one-layer CST curvature trial."""

    atoms: int = 1
    amplitude: float = 1e-3
    learning_rate: float = 1e-2
    separation: float = 0.16
    sigma: float = 0.28
    n_in: int = 5
    n_out: int = 4
    batch_size: int = 8
    seed: int = 17

    def __post_init__(self) -> None:
        if self.atoms <= 0:
            raise ValueError("atoms must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.separation < 0:
            raise ValueError("separation must be non-negative")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.n_in <= 1 or self.n_out <= 1:
            raise ValueError("n_in and n_out must exceed one")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class TrialResult:
    """Predictions and exact remainders for one SGD update."""

    atoms: int
    amplitude: float
    learning_rate: float
    initial_loss: float
    updated_loss: float
    gradient_norm: float
    step_norm: float
    amplitude_step_norm: float
    position_step_norm: float
    p1: float
    p2_cst: float
    p2_full: float
    p_weight_exact: float
    p_actual: float
    cst_quadratic: float
    cst_quadratic_amplitude_amplitude: float
    cst_quadratic_amplitude_position: float
    cst_quadratic_position_position: float
    cst_higher_order: float
    loss_quadratic_jacobian: float
    loss_quadratic_exact: float
    cross_atom_map_hessian_max: float

    def relative_error(self, prediction: str) -> float:
        """Symmetric relative error against the actual finite loss change."""
        if prediction not in {"p1", "p2_cst", "p2_full", "p_weight_exact"}:
            raise ValueError(f"unknown prediction {prediction!r}")
        predicted = getattr(self, prediction)
        scale = max(abs(predicted), abs(self.p_actual), 1e-15)
        return abs(predicted - self.p_actual) / scale


def initial_theta(config: TrialConfig) -> Tensor:
    """Return rows ``(w, s, t)`` with controlled atom overlap."""
    dtype = torch.float64
    offsets = (
        torch.arange(config.atoms, dtype=dtype) - (config.atoms - 1.0) / 2.0
    ) * config.separation
    source = 0.43 + offsets
    target = 0.57 - offsets
    amplitude = torch.full((config.atoms,), config.amplitude, dtype=dtype)
    return torch.stack((amplitude, source, target), dim=1)


def charts(config: TrialConfig) -> tuple[Tensor, Tensor]:
    """Fixed one-dimensional input and output neuron charts."""
    return (
        torch.linspace(0.0, 1.0, config.n_in, dtype=torch.float64)[:, None],
        torch.linspace(0.0, 1.0, config.n_out, dtype=torch.float64)[:, None],
    )


def gaussian_cst_weight(
    theta: Tensor,
    mu_in: Tensor,
    mu_out: Tensor,
    factor: GaussianFactor,
) -> Tensor:
    """Materialize ``sum_k w_k kout_k kin_k.T`` from ``(w, s, t)`` rows."""
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("theta must have shape [atoms, 3] for (w, s, t)")
    weights = theta[:, 0]
    source = theta[:, 1:2]
    target = theta[:, 2:3]
    k_in = factor(mu_in, source)
    k_out = factor(mu_out, target)
    return (k_out * weights[None, :]) @ k_in.transpose(0, 1)


def _data(config: TrialConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    inputs = torch.randn(
        config.batch_size,
        config.n_in,
        generator=generator,
        dtype=torch.float64,
    )
    teacher = (
        torch.randn(
            config.n_out,
            config.n_in,
            generator=generator,
            dtype=torch.float64,
        )
        / config.n_in**0.5
    )
    noise = 0.03 * torch.randn(
        config.batch_size,
        config.n_out,
        generator=generator,
        dtype=torch.float64,
    )
    targets = F.linear(inputs, teacher) + noise
    return inputs, targets


def _contracted_map_hessian(
    weight_fn,
    theta: Tensor,
    weight_gradient: Tensor,
) -> Tensor:
    """Hessian of ``<grad_W L, W(theta)>`` with ``grad_W L`` frozen."""

    def frozen_weight_loss(point: Tensor) -> Tensor:
        return (weight_gradient * weight_fn(point)).sum()

    return torch.autograd.functional.hessian(frozen_weight_loss, theta)


def _cross_atom_maximum(hessian: Tensor) -> Tensor:
    atoms = hessian.shape[0]
    if atoms == 1:
        return hessian.new_zeros(())
    mask = ~torch.eye(atoms, dtype=torch.bool, device=hessian.device)
    cross_blocks = hessian.permute(0, 2, 1, 3)[mask]
    return cross_blocks.abs().amax()


def _same_atom_quadratic_parts(
    hessian: Tensor, step: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``1/2 step.T H step`` into within-atom field blocks."""
    blocks = hessian.permute(0, 2, 1, 3)
    amplitude_amplitude = hessian.new_zeros(())
    amplitude_position = hessian.new_zeros(())
    position_position = hessian.new_zeros(())
    for atom in range(step.shape[0]):
        block = blocks[atom, atom]
        dw = step[atom, 0]
        dp = step[atom, 1:]
        amplitude_amplitude = amplitude_amplitude + 0.5 * dw * block[0, 0] * dw
        # The factor 1/2 cancels because the symmetric w-p and p-w entries
        # both occur in step.T H step.
        amplitude_position = amplitude_position + dw * (block[0, 1:] @ dp)
        position_position = position_position + 0.5 * dp @ block[1:, 1:] @ dp
    return amplitude_amplitude, amplitude_position, position_position


def run_trial(config: TrialConfig) -> TrialResult:
    """Measure four approximations along one ordinary SGD update."""
    theta = initial_theta(config)
    mu_in, mu_out = charts(config)
    inputs, targets = _data(config)
    factor = GaussianFactor(config.sigma, learnable=False).double()

    def weight_fn(point: Tensor) -> Tensor:
        return gaussian_cst_weight(point, mu_in, mu_out, factor)

    def loss_from_weight(weight: Tensor) -> Tensor:
        return F.mse_loss(F.linear(inputs, weight), targets)

    def loss_fn(point: Tensor) -> Tensor:
        return loss_from_weight(weight_fn(point))

    theta_for_grad = theta.detach().requires_grad_(True)
    initial_loss = loss_fn(theta_for_grad)
    gradient = torch.autograd.grad(initial_loss, theta_for_grad)[0].detach()
    step = -config.learning_rate * gradient

    weight = weight_fn(theta).detach()
    weight_for_grad = weight.detach().requires_grad_(True)
    weight_loss = loss_from_weight(weight_for_grad)
    weight_gradient = torch.autograd.grad(weight_loss, weight_for_grad)[0].detach()

    def first_map_direction(point: Tensor) -> Tensor:
        return torch.func.jvp(weight_fn, (point,), (step,))[1]

    jacobian_step = first_map_direction(theta)
    second_map_direction = torch.func.jvp(first_map_direction, (theta,), (step,))[1]
    updated_weight = weight_fn(theta + step).detach()
    exact_weight_change = updated_weight - weight

    p1 = (weight_gradient * jacobian_step).sum()
    cst_quadratic = 0.5 * (weight_gradient * second_map_direction).sum()
    p2_cst = p1 + cst_quadratic
    p_weight_exact = (weight_gradient * exact_weight_change).sum()

    # For MSE, 1/2 delta_W^T H_W delta_W is exactly the mean squared output
    # change.  The Jacobian variant is the H_W term in the full theta-space
    # second-order Taylor expansion; the exact variant uses the actual CST
    # finite weight change and closes P_weight_exact to P_actual exactly.
    jacobian_output_change = F.linear(inputs, jacobian_step)
    exact_output_change = F.linear(inputs, exact_weight_change)
    loss_quadratic_jacobian = jacobian_output_change.square().mean()
    loss_quadratic_exact = exact_output_change.square().mean()
    p2_full = p2_cst + loss_quadratic_jacobian

    updated_loss = loss_from_weight(updated_weight)
    p_actual = updated_loss - initial_loss.detach()
    map_hessian = _contracted_map_hessian(weight_fn, theta, weight_gradient)
    (
        cst_amplitude_amplitude,
        cst_amplitude_position,
        cst_position_position,
    ) = _same_atom_quadratic_parts(map_hessian, step)

    # Chain-rule agreement: the ordinary theta gradient is J^T grad_W L.
    p1_from_theta = (gradient * step).sum()
    torch.testing.assert_close(p1, p1_from_theta, rtol=1e-10, atol=1e-12)
    # MSE has no W-space Taylor remainder above second order.
    torch.testing.assert_close(
        p_actual,
        p_weight_exact + loss_quadratic_exact,
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        cst_quadratic,
        cst_amplitude_amplitude + cst_amplitude_position + cst_position_position,
        rtol=1e-10,
        atol=1e-12,
    )

    def number(value: Tensor) -> float:
        return float(value.detach())

    return TrialResult(
        atoms=config.atoms,
        amplitude=config.amplitude,
        learning_rate=config.learning_rate,
        initial_loss=number(initial_loss),
        updated_loss=number(updated_loss),
        gradient_norm=number(torch.linalg.vector_norm(gradient)),
        step_norm=number(torch.linalg.vector_norm(step)),
        amplitude_step_norm=number(torch.linalg.vector_norm(step[:, 0])),
        position_step_norm=number(torch.linalg.vector_norm(step[:, 1:])),
        p1=number(p1),
        p2_cst=number(p2_cst),
        p2_full=number(p2_full),
        p_weight_exact=number(p_weight_exact),
        p_actual=number(p_actual),
        cst_quadratic=number(cst_quadratic),
        cst_quadratic_amplitude_amplitude=number(cst_amplitude_amplitude),
        cst_quadratic_amplitude_position=number(cst_amplitude_position),
        cst_quadratic_position_position=number(cst_position_position),
        cst_higher_order=number(p_weight_exact - p2_cst),
        loss_quadratic_jacobian=number(loss_quadratic_jacobian),
        loss_quadratic_exact=number(loss_quadratic_exact),
        cross_atom_map_hessian_max=number(_cross_atom_maximum(map_hessian)),
    )


def run_sweep(
    *,
    atoms: Iterable[int] = (1, 2),
    amplitudes: Iterable[float] = (0.0, 1e-6, 1e-3, 1e-1, 1.0),
    learning_rates: Iterable[float] = (1e-3, 1e-2, 1e-1),
    **common,
) -> list[TrialResult]:
    """Run the Cartesian product of the principal experiment controls."""
    return [
        run_trial(
            TrialConfig(
                atoms=atom_count,
                amplitude=amplitude,
                learning_rate=learning_rate,
                **common,
            )
        )
        for atom_count, amplitude, learning_rate in product(
            atoms, amplitudes, learning_rates
        )
    ]


def _format_table(results: Iterable[TrialResult]) -> str:
    header = (
        "atoms amp lr actual p1 p2_cst W_exact "
        "err_p1 err_p2 err_W Hwp2 Hpp2 HW2(Jd) cross_map_H"
    )
    rows = [header]
    for result in results:
        rows.append(
            f"{result.atoms:>5d} "
            f"{result.amplitude:>8.1e} "
            f"{result.learning_rate:>8.1e} "
            f"{result.p_actual:>11.3e} "
            f"{result.p1:>11.3e} "
            f"{result.p2_cst:>11.3e} "
            f"{result.p_weight_exact:>11.3e} "
            f"{result.relative_error('p1'):>8.2e} "
            f"{result.relative_error('p2_cst'):>8.2e} "
            f"{result.relative_error('p_weight_exact'):>8.2e} "
            f"{result.cst_quadratic_amplitude_position:>9.2e} "
            f"{result.cst_quadratic_position_position:>9.2e} "
            f"{result.loss_quadratic_jacobian:>9.2e} "
            f"{result.cross_atom_map_hessian_max:>11.2e}"
        )
    return "\n".join(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-3, 1e-1, 1.0],
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--separation", type=float, default=0.16)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = _parser().parse_args()
    common = {
        field.name: getattr(args, field.name)
        for field in fields(TrialConfig)
        if hasattr(args, field.name)
        and field.name not in {"atoms", "amplitude", "learning_rate"}
    }
    results = run_sweep(
        atoms=args.atoms,
        amplitudes=args.amplitudes,
        learning_rates=args.learning_rates,
        **common,
    )
    print(_format_table(results))


if __name__ == "__main__":
    main()
