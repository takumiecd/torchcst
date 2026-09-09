"""Public configuration values for model-level CST optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from torchcst.atoms import AtomGradMode

from .solvers import FullQuartic, QuarticSolver


@dataclass(frozen=True)
class FirstOrderAdamConfig:
    """Tangent-only Adam; block/diagonal visible RMS are experimental metrics."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    trust_radius: float = 0.25
    second_moment: Literal["separable", "atom_block", "atom_diag"] = "separable"
    first_moment_damping: float = 0.0
    atom_grad_mode: AtomGradMode = "auto"
    row_chunk_size: int = 16
    factored_geometry: bool = True
    tangent_rtol: float = 1e-6
    tangent_backend: Literal["auto", "specialized", "factor_autograd", "reference"] = (
        "auto"
    )
    tangent_atom_tile: int = 32
    recompression: Literal["direct", "pcg"] = "direct"
    recompression_action: Literal["pair", "jvp_vjp"] = "pair"
    recompression_max_iter: int = 64
    recompression_rtol: float = 1e-5

    device_execution: bool = False
    update_solver: Literal["spectral", "pcg", "krylov"] = "spectral"
    update_approximation: Literal["full", "diagonal", "atom_block"] = "full"
    update_max_iter: int = 512
    update_shift_steps: int = 32
    update_rtol: float = 1e-5
    update_basis_size: int = 128
    update_basis_memory_mb: float = 64.0
    update_check_interval: int = 32

    def __post_init__(self) -> None:
        from torchcst._derivatives.tangent_solve import validate_pcg

        if self.update_solver not in ("spectral", "pcg", "krylov"):
            raise ValueError("update_solver must be spectral, pcg or krylov")
        if (
            not math.isfinite(self.update_basis_memory_mb)
            or self.update_basis_memory_mb <= 0
        ):
            raise ValueError("update_basis_memory_mb must be finite and positive")
        if self.update_approximation not in ("full", "diagonal", "atom_block"):
            raise ValueError(
                "update_approximation must be full, diagonal or atom_block"
            )
        if self.update_approximation != "full" and (
            self.update_solver != "spectral"
            or self.second_moment != "separable"
            or not self.factored_geometry
            or self.tangent_backend == "reference"
        ):
            raise ValueError(
                "local update approximations require spectral solver, separable "
                "moments and factorized tangents"
            )
        for name in (
            "update_max_iter",
            "update_shift_steps",
            "update_basis_size",
            "update_check_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.update_rtol) or not 0 < self.update_rtol < 1:
            raise ValueError("update_rtol must be finite and between zero and one")
        if self.update_solver in ("pcg", "krylov") and (
            self.second_moment != "separable"
            or not self.factored_geometry
            or self.tangent_backend == "reference"
        ):
            raise ValueError(
                "Matrix-free update requires separable moments and factorized tangents"
            )

        if not isinstance(self.device_execution, bool):
            raise TypeError("device_execution must be a bool")
        if self.device_execution and (
            self.recompression != "pcg"
            or self.second_moment != "separable"
            or not self.factored_geometry
            or self.tangent_backend == "reference"
        ):
            raise ValueError(
                "device_execution requires PCG, separable moments and factorized tangents"
            )

        if self.tangent_backend not in (
            "auto",
            "specialized",
            "factor_autograd",
            "reference",
        ):
            raise ValueError("invalid tangent_backend")
        if not self.factored_geometry and self.tangent_backend not in (
            "auto",
            "reference",
        ):
            raise ValueError(
                "factored_geometry=False requires auto or reference backend"
            )
        if (
            isinstance(self.tangent_atom_tile, bool)
            or not isinstance(self.tangent_atom_tile, int)
            or self.tangent_atom_tile < 1
        ):
            raise ValueError("tangent_atom_tile must be a positive integer")
        if self.recompression_action not in ("pair", "jvp_vjp"):
            raise ValueError("recompression_action must be pair or jvp_vjp")
        if self.recompression_action == "jvp_vjp" and (
            self.recompression != "pcg"
            or not self.factored_geometry
            or self.tangent_backend == "reference"
        ):
            raise ValueError(
                "jvp_vjp recompression requires PCG and factorized tangents"
            )
        if self.recompression not in ("direct", "pcg"):
            raise ValueError("recompression must be direct or pcg")
        validate_pcg(
            damping=self.first_moment_damping if self.recompression == "pcg" else 1.0,
            max_iter=self.recompression_max_iter,
            rtol=self.recompression_rtol,
        )
        for name in ("lr", "eps", "trust_radius", "tangent_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tangent_rtol >= 1:
            raise ValueError("tangent_rtol must be less than one")
        if (
            not math.isfinite(self.first_moment_damping)
            or self.first_moment_damping < 0
        ):
            raise ValueError("first_moment_damping must be finite and nonnegative")
        object.__setattr__(self, "betas", _validate_betas(self.betas))
        if self.second_moment not in ("separable", "atom_block", "atom_diag"):
            raise ValueError("unknown first-order second_moment")
        if self.atom_grad_mode not in ("auto", "custom", "hooks"):
            raise ValueError("invalid atom_grad_mode")
        if not isinstance(self.factored_geometry, bool):
            raise TypeError("factored_geometry must be a bool")
        if isinstance(self.row_chunk_size, bool) or not isinstance(
            self.row_chunk_size, int
        ):
            raise TypeError("row_chunk_size must be an integer")
        if self.row_chunk_size < 1:
            raise ValueError("row_chunk_size must be positive")


def _validate_betas(betas: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("betas must be a pair")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("betas must satisfy 0 <= beta < 1")
    return float(beta1), float(beta2)


@dataclass(frozen=True)
class SecondOrderAdamConfig:
    """Configuration of the compact implicit optimizer for every CST site."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    second_moment: Literal["separable"] = "separable"
    trust_radius: float = 0.25
    quartic: QuarticSolver = field(default_factory=FullQuartic)
    quartic_evaluation: Literal["auto", "visible", "gram"] = "auto"
    atom_grad_mode: AtomGradMode = "auto"
    row_chunk_size: int = 64
    first_moment_damping: float = 0.0
    device_execution: bool = False
    factored_geometry: bool = False
    gram_solver: Literal["jacobi", "cholesky", "pcg"] = "jacobi"

    gram_iterations: int = 64
    gram_rtol: float = 1e-5
    gram_block_size: int = 32

    def __post_init__(self) -> None:
        from torchcst._derivatives._pcg import PCGOptions

        PCGOptions(self.gram_iterations, self.gram_rtol, self.gram_block_size)
        if self.gram_solver not in ("jacobi", "cholesky", "pcg"):
            raise ValueError("gram_solver must be jacobi, cholesky, or pcg")
        if self.gram_solver == "cholesky" and self.first_moment_damping <= 0:
            raise ValueError("cholesky requires positive first_moment_damping")
        if self.gram_solver == "pcg":
            if not self.factored_geometry:
                raise ValueError("pcg requires factored_geometry")
            if (
                not math.isfinite(self.first_moment_damping)
                or self.first_moment_damping <= 0
            ):
                raise ValueError("pcg requires finite positive first_moment_damping")
        if not isinstance(self.factored_geometry, bool):
            raise TypeError("factored_geometry must be a bool")
        if not isinstance(self.device_execution, bool):
            raise TypeError("device_execution must be a bool")
        if self.device_execution and not getattr(
            self.quartic, "supports_deferred_execution", False
        ):
            raise ValueError(
                "device_execution requires a solver with deferred device diagnostics"
            )
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
        if self.quartic_evaluation not in ("auto", "visible", "gram"):
            raise ValueError("quartic_evaluation must be 'auto', 'visible', or 'gram'")
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


@dataclass(frozen=True)
class LocalAdamConfig:
    """Atom-local transported moments and direct regularized updates.

    Damping is added to each local matrix before solving. No trust radius,
    iterative solver, or cross-atom moment coupling is used.
    """

    lr: float = 0.05
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    first_moment_damping: float = 1e-2
    update_damping: float = 1e-2
    tangent_rtol: float = 1e-6
    solve_rtol: float = 1e-5
    atom_grad_mode: AtomGradMode = "auto"
    row_chunk_size: int = 16
    factored_geometry: bool = True
    tangent_backend: Literal["auto", "specialized", "factor_autograd", "reference"] = (
        "auto"
    )
    tangent_atom_tile: int = 32
    device_execution: bool = False

    whitening: Literal["eigen", "cholesky"] = "cholesky"
    whitening_damping: float | None = 1e-4

    def __post_init__(self):
        if self.whitening not in ("eigen", "cholesky"):
            raise ValueError("whitening must be eigen or cholesky")
        if self.whitening_damping is not None and (
            not math.isfinite(self.whitening_damping) or self.whitening_damping <= 0
        ):
            raise ValueError("whitening_damping must be finite and positive or None")
        # Reuse the common tangent/observation validation without exposing its
        # unrelated solver choices in this configuration.
        FirstOrderAdamConfig(
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            first_moment_damping=self.first_moment_damping,
            tangent_rtol=self.tangent_rtol,
            atom_grad_mode=self.atom_grad_mode,
            row_chunk_size=self.row_chunk_size,
            factored_geometry=self.factored_geometry,
            tangent_backend=self.tangent_backend,
            tangent_atom_tile=self.tangent_atom_tile,
        )
        object.__setattr__(self, "betas", _validate_betas(self.betas))
        for name in ("first_moment_damping", "update_damping", "solve_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.solve_rtol >= 1:
            raise ValueError("solve_rtol must be less than one")
        if not isinstance(self.device_execution, bool):
            raise TypeError("device_execution must be a bool")
        if self.device_execution and (
            not self.factored_geometry or self.tangent_backend == "reference"
        ):
            raise ValueError("device_execution requires factorized tangents")
