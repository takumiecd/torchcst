"""Public configuration values for the retained CST optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from torchcst.atoms import AtomGradMode

from .atom_grad import CurvatureBlockMask
from .solvers import (
    NormalizedFixedPointSolver,
    NormalizedSolver,
    QuadraticGradientSolver,
)


def _validate_betas(betas: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("betas must be a pair")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("betas must satisfy 0 <= beta < 1")
    return float(beta1), float(beta2)


def _validate_nd_config(config) -> None:
    for name in ("lr", "eps", "trust_radius"):
        value = getattr(config, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    object.__setattr__(config, "betas", _validate_betas(config.betas))
    if not isinstance(config.solver, NormalizedSolver):
        raise TypeError("solver must be a NormalizedSolver")
    if not isinstance(config.initial_zero_step, bool):
        raise TypeError("initial_zero_step must be a bool")
    if config.atom_grad_mode not in ("auto", "custom", "hooks"):
        raise ValueError("atom_grad_mode must be 'auto', 'custom', or 'hooks'")
    if not isinstance(config.factored, bool):
        raise TypeError("factored must be a bool")
    if not isinstance(config.device_execution, bool):
        raise TypeError("device_execution must be a bool")
    if config.curvature_mask is not None and not isinstance(
        config.curvature_mask, CurvatureBlockMask
    ):
        raise TypeError("curvature_mask must be a CurvatureBlockMask or None")
    if config.kernel_step_size is not None and (
        isinstance(config.kernel_step_size, bool)
        or not math.isfinite(config.kernel_step_size)
        or config.kernel_step_size <= 0
    ):
        raise ValueError("kernel_step_size must be finite and positive or None")


@dataclass(frozen=True)
class _NDOptimizerConfig:
    """Shared fields for composable N/D optimizer families."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    trust_radius: float = 0.1
    solver: NormalizedSolver = field(default_factory=NormalizedFixedPointSolver)
    initial_zero_step: bool = False
    atom_grad_mode: AtomGradMode = "auto"
    factored: bool = False
    device_execution: bool = False
    kernel_step_size: float | None = None
    curvature_mask: CurvatureBlockMask | None = None

    def __post_init__(self) -> None:
        _validate_nd_config(self)


@dataclass(frozen=True)
class NormalizedOptimizerConfig(_NDOptimizerConfig):
    """Implicit family: ``d ← Π(-η N(d)/D(d))``."""


@dataclass(frozen=True)
class QuadraticOptimizerConfig(_NDOptimizerConfig):
    """Gradient family: ``d ← Π(d - η N(d)/D(d))``."""

    solver: NormalizedSolver = field(default_factory=QuadraticGradientSolver)


@dataclass(frozen=True)
class AdamRConfig(_NDOptimizerConfig):
    """Normalized or Quadratic N/D Adam plus decoupled repulsion."""

    solver: NormalizedSolver | None = None
    update_rule: Literal["normalized", "quadratic"] = "normalized"
    repulsion: float = 0.0
    kind: Literal["cosine", "raw", "abs"] = "cosine"

    def __post_init__(self) -> None:
        if self.update_rule not in ("normalized", "quadratic"):
            raise ValueError("update_rule must be 'normalized' or 'quadratic'")
        if self.solver is None:
            solver = (
                NormalizedFixedPointSolver()
                if self.update_rule == "normalized"
                else QuadraticGradientSolver()
            )
            object.__setattr__(self, "solver", solver)
        super().__post_init__()
        if self.solver.update_rule != self.update_rule:
            raise ValueError("solver update rule does not match update_rule")
        if (
            isinstance(self.repulsion, bool)
            or not math.isfinite(self.repulsion)
            or self.repulsion < 0
        ):
            raise ValueError("repulsion must be finite and nonnegative")
        if self.kind not in ("cosine", "raw", "abs"):
            raise ValueError("kind must be 'cosine', 'raw', or 'abs'")


@dataclass(frozen=True)
class AdamWConfig:
    """Configuration of the functional dense AdamW block."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        object.__setattr__(self, "betas", _validate_betas(self.betas))
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
