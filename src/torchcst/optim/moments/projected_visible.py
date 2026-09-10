"""Atom-local compression of a visible diagonal second-moment operator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchcst._runtime.validation import require

from ..atom_grad import AtomGradRequest
from .atom_rms import AtomBlockMetric, symmetric_root
from .base import ExpandedSecondMoment, SecondMomentComponent


@dataclass(frozen=True)
class ProjectedVisibleSecondMomentState:
    """Coefficients of ``J @ coefficient @ J.T`` at the saved point."""

    point: torch.Tensor
    coefficient: torch.Tensor
    beta_power: float


class ProjectedVisibleSecondMoment(SecondMomentComponent):
    r"""Project dense diagonal-Adam evidence into independent atom blocks.

    If ``U = J.T @ Diag(v) @ J`` and ``R = J.T @ J``, the stored coefficient
    solves ``(R + damping I) @ Gamma @ (R + damping I) = U``.  It therefore
    represents the atom-local ambient operator ``J @ Gamma @ J.T`` without a
    whitening-basis state.
    """

    def __init__(self, beta: float, *, eps: float, damping: float, rtol: float):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and nonnegative")
        if not math.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("rtol must be finite and between zero and one")
        self.beta = float(beta)
        self.eps = float(eps)
        self.damping = float(damping)
        self.rtol = float(rtol)

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(atom_square=True)

    def initialize(self, context) -> ProjectedVisibleSecondMomentState:
        point = context.current_point
        k, p = point.shape
        return ProjectedVisibleSecondMomentState(
            point=point.clone(),
            coefficient=point.new_zeros(k, p, p),
            beta_power=1.0,
        )

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, ProjectedVisibleSecondMomentState):
            raise TypeError("expected a projected visible second-moment state")
        if not math.isclose(state.beta_power, self.beta ** (next_step - 1)):
            raise ValueError("projected visible second-moment step or beta mismatch")
        observation.require(self.observation_request)
        assert observation.atom_square is not None

        point, geometry = context.current_point, context.geometry
        k, p = point.shape
        if observation.atom_square.shape != (k, p, p):
            raise ValueError("atom-square observation has the wrong shape")

        gram = geometry.cross(point, point, local=True).double()
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        eye = torch.eye(p, device=point.device, dtype=gram.dtype)
        regularized = gram + self.damping * eye
        factor, info = torch.linalg.cholesky_ex(regularized, check_errors=False)
        require(
            (info == 0).all(),
            "projected visible second-moment Cholesky failed",
            FloatingPointError,
        )

        if next_step == 1:
            transported = torch.zeros_like(gram)
        else:
            cross = geometry.cross(point, state.point, local=True).double()
            transported = (
                cross
                @ state.coefficient.double()
                @ cross.transpose(-1, -2)
            )
        numerator = self.beta * transported + (
            1.0 - self.beta
        ) * observation.atom_square.double()
        numerator = 0.5 * (numerator + numerator.transpose(-1, -2))

        left = torch.cholesky_solve(numerator, factor)
        coefficient = torch.cholesky_solve(
            left.transpose(-1, -2), factor
        ).transpose(-1, -2)
        residual = regularized @ coefficient @ regularized - numerator
        relative = residual.norm(dim=(-2, -1)) / numerator.norm(
            dim=(-2, -1)
        ).clamp_min(1e-300)
        require(
            torch.isfinite(coefficient).all() & (relative <= self.rtol).all(),
            "projected visible second-moment compression failed",
            FloatingPointError,
        )

        beta_power = state.beta_power * self.beta
        corrected = numerator / (1.0 - beta_power)
        normalized = torch.linalg.solve_triangular(
            factor, corrected, upper=False
        )
        normalized = torch.linalg.solve_triangular(
            factor, normalized.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        root = symmetric_root(normalized)
        metric = factor @ (root + self.eps * eye) @ factor.transpose(-1, -2)
        require(
            torch.isfinite(metric).all(),
            "projected visible second-moment metric is non-finite",
            FloatingPointError,
        )

        pending = ProjectedVisibleSecondMomentState(
            point=point.clone(),
            coefficient=coefficient.to(point).detach(),
            beta_power=beta_power,
        )
        return ExpandedSecondMoment(
            AtomBlockMetric(metric.detach(), geometry.visible_shape), pending
        )

    def compress(self, expanded, accepted_displacement, context):
        del accepted_displacement, context
        pending = expanded.pending_state
        if not isinstance(pending, ProjectedVisibleSecondMomentState):
            raise TypeError("expanded second moment belongs to another component")
        return ProjectedVisibleSecondMomentState(
            point=pending.point.clone(),
            coefficient=pending.coefficient.clone(),
            beta_power=pending.beta_power,
        )
