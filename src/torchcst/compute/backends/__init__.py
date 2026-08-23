"""Execution backends: the ways one CST measure can be applied.

A compute module owns state (stores, views, capture, mass); a backend is
the pure-function world it dispatches into.  Backend code receives plain
tensors -- coordinates, amplitudes, kernel columns, rows -- and returns
tensors.  It never imports stores, modules, or the engine, so the
dependency arrow has one direction: ``modules -> backends -> torch``.

Four ways to deliver ``W = K_out diag(w) K_in^T`` applied to rows:

* :class:`Factored` -- no W: ``((x @ k_in) * w) @ k_out.T``.  Per row it
  costs ``K (d_in + d_out)`` FLOPs and autograd retains ``[rows, K]``;
  the right form only below the crossover.
* :class:`Materialized` -- build W once per forward, apply one
  cuBLAS GEMM / cuDNN conv.  Ceiling = dense speed; ``lean=True`` swaps
  the build's autograd for a closed-form chunked backward so peak memory
  stays O(chunk) at any K.
* :class:`NativeTruncated` -- no W and no ``[rows, K]``: kernels
  truncated at ``radius`` sigma, rows routed through per-atom neighbor
  tables.  ``rows x K x (m_in + m_out)`` FLOPs -- under a dense GEMM in
  the lawful 20-30 sigma domain, which neither other backend can be.
  The PyTorch implementation here is the semantics oracle for the fused
  (Triton) kernel that realizes the FLOP advantage.
* ``"auto"`` -- the module picks Factored or Materialized per forward
  from the live atom count via :func:`crossover_materializes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchcst.representation import Amplitude, L2NormalizedColumns


@dataclass(frozen=True)
class Factored:
    """Apply the factored form directly; no dense weight exists."""


@dataclass(frozen=True)
class Materialized:
    """Build the dense weight per forward and lean on cuBLAS/cuDNN.

    ``lean`` replaces the build's default autograd (which retains
    ``[features, K, d]`` kernel broadcasts) with a closed-form chunked
    backward saving only atom parameters -- Gaussian kernels and
    ``track_mass=False`` required.  ``compute_dtype`` runs the build's
    contraction in reduced precision (tensor cores accumulate fp32);
    parameters, the GEMM/conv, and gradients keep the parameter dtype.
    """

    lean: bool = False
    compute_dtype: torch.dtype | None = None
    compile_l2: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lean, bool):
            raise TypeError("lean must be a bool")
        if not isinstance(self.compile_l2, bool):
            raise TypeError("compile_l2 must be a bool")
        if self.compile_l2 and not self.lean:
            raise ValueError("compile_l2 requires lean=True")
        if self.compute_dtype is not None and (
            not isinstance(self.compute_dtype, torch.dtype)
            or not self.compute_dtype.is_floating_point
        ):
            raise TypeError("compute_dtype must be a floating dtype or None")


@dataclass(frozen=True)
class NativeTruncated:
    """Truncate kernels at ``radius`` sigma; apply via neighbor tables.

    The support boundary follows retraction semantics -- zero position
    gradient outside, exactly like the displacement-box clamp on
    ``OffsetCSTConv2d`` -- and the dropped tail is ``exp(-radius^2/2)``
    per matrix entry relative to the atom's peak.  Gaussian kernels and
    ``track_mass=False`` required (closed-form backward; mass's kernel
    matrices would resurrect the memory this backend removes).
    """

    radius: float

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(
            self.radius, (int, float)
        ):
            raise TypeError("radius must be a number")
        if not self.radius > 0.0:
            raise ValueError("radius must be positive")


Backend = Factored | Materialized | NativeTruncated


def validate_backend(backend, *, kernel_in, kernel_out, track_mass, gauge=None):
    """Check a module's backend choice against its site; return it.

    The closed-form backends (lean materialization, native truncated)
    hard-require Gaussian kernels (analytic derivatives) and
    ``track_mass=False`` (mass would rebuild the full kernel matrices).
    Lean linear materialisation also supports unit-L2 Gaussian columns: their
    norms and tangent-projected derivatives are computed one chunk at a time.
    Native truncation still requires the amplitude gauge because an exact
    normalising gauge needs the complete feature column.
    """
    if backend == "auto":
        return backend
    if not isinstance(backend, (Factored, Materialized, NativeTruncated)):
        raise TypeError(
            'backend must be "auto", Factored, Materialized, or '
            "NativeTruncated"
        )
    closed_form = isinstance(backend, NativeTruncated) or (
        isinstance(backend, Materialized) and backend.lean
    )
    if closed_form:
        name = type(backend).__name__
        if kernel_in.family != "gaussian" or kernel_out.family != "gaussian":
            raise ValueError(f"{name} requires Gaussian kernels")
        if track_mass:
            raise ValueError(f"{name} requires track_mass=False")
        allowed_gauge = isinstance(gauge, Amplitude) or (
            isinstance(backend, Materialized)
            and backend.lean
            and isinstance(gauge, L2NormalizedColumns)
        )
        if gauge is not None and not allowed_gauge:
            raise ValueError(
                f"{name} requires the Amplitude gauge; exact normalisation "
                "is only available in lean linear materialisation"
            )
        if isinstance(backend, Materialized) and backend.compile_l2 and not isinstance(
            gauge, L2NormalizedColumns
        ):
            raise ValueError("compile_l2 requires the L2NormalizedColumns gauge")
    return backend


def crossover_materializes(count: int, d_in: int, d_out: int) -> bool:
    """The auto rule: materialize past the FLOP crossover.

    Factored costs ``count * (d_in + d_out)`` per row; the materialized
    apply costs ``d_in * d_out``.
    """
    return count * (d_in + d_out) > d_in * d_out


__all__ = [
    "Backend",
    "Factored",
    "Materialized",
    "NativeTruncated",
    "crossover_materializes",
    "validate_backend",
]
