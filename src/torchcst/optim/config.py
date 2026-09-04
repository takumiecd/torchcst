"""Public configuration values for model-level CST optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torchcst.atoms import AtomGradMode

from .solvers import FullQuartic, QuarticSolver


def _validate_betas(betas: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("betas must be a pair")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("betas must satisfy 0 <= beta < 1")
    return float(beta1), float(beta2)


@dataclass(frozen=True)
class ImplicitAdamConfig:
    """Configuration of the compact implicit optimizer for every CST site."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    second_moment: Literal["separable"] = "separable"
    trust_radius: float = 0.25
    quartic: QuarticSolver = field(default_factory=FullQuartic)
    atom_grad_mode: AtomGradMode = "auto"
    row_chunk_size: int = 64
    first_moment_damping: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        object.__setattr__(self, "betas", _validate_betas(self.betas))
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.second_moment != "separable":
            raise ValueError("second_moment must be 'separable'")
        if self.trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        if not isinstance(self.quartic, QuarticSolver):
            raise TypeError("quartic must be a QuarticSolver")
        if self.atom_grad_mode not in ("auto", "custom", "hooks"):
            raise ValueError("atom_grad_mode must be 'auto', 'custom', or 'hooks'")
        if isinstance(self.row_chunk_size, bool) or not isinstance(
            self.row_chunk_size, int
        ):
            raise TypeError("row_chunk_size must be an integer")
        if self.row_chunk_size < 1:
            raise ValueError("row_chunk_size must be positive")
        if self.first_moment_damping < 0:
            raise ValueError("first_moment_damping must be nonnegative")


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
