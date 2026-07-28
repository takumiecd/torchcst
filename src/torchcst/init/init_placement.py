"""Stage 4 of ``docs/absorb-and-gram-design.md``: initialization as birth.

Pure, engine-free functions that compute *where* to place ``K`` synapse
atoms and *with what amplitudes*, returning plain tensors plus a small
helper that wraps them into a ``list[SynapseBirth]`` ready for
``SynapseStore.apply``/``SynapseStore.prepare`` -- exactly the shape used by
``examples/gaussian_gradient_birth.py``. Initialization is birth on the
empty state: there is no backward-derived score to consult yet, so these
functions depend only on geometry (:func:`coverage_lattice`), the same
D-weighted marginal-gain quantity ``GramService`` uses for absorb
(:func:`logdet_greedy`), and a variance-preservation rule for amplitudes
(:func:`variance_amplitudes`).

This module imports only ``torch`` and
:class:`torchcst.storage.synapse.SynapseBirth` (for the wrapping helper). It
does not import ``torchcst.engine``, ``torchcst.policy``,
``torchcst.instruments``, ``torchcst.compute``, or
``torchcst.representation.gram`` -- callers obtain kernel columns
themselves (e.g. via ``KernelPort.columns`` or a compute module's
``kernel_columns``) and pass them in.

Gram convention (matches :class:`torchcst.representation.gram.GramService`
exactly, restated here since this module must not import it): for atoms
with output-side kernel columns ``U`` (``[n_out, N]``) and input-side kernel
columns ``V`` (``[n_in, N]``), the D-weighted Gram is assembled separably

    Gamma_D[i, j] = (U[:, i] . U[:, j]) * (V[:, i]^T Sigma_x V[:, j])

(a Hadamard product of two rank-``N`` Grams), with ``Sigma_x = X^T X / n``
for an optional data batch ``X`` and the identity metric when no data is
given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from torchcst.storage.synapse import SynapseBirth

__all__ = [
    "GreedySelection",
    "coverage_lattice",
    "logdet_greedy",
    "variance_amplitudes",
    "to_synapse_births",
]


# ---------------------------------------------------------------------------
# Function 1: coverage_lattice
# ---------------------------------------------------------------------------


def _as_axis_tuple(value: float | Sequence[float], dim: int, name: str) -> tuple[float, ...]:
    """Mirror :class:`torchcst.representation.Box`'s scalar-or-per-axis convention."""
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return (float(value),) * dim
        if value.ndim != 1 or value.numel() != dim:
            raise ValueError(f"{name} must be a scalar or have {dim} entries")
        return tuple(float(item) for item in value.detach().cpu())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),) * dim
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != dim:
            raise ValueError(f"{name} must be a scalar or have {dim} entries")
        return tuple(float(item) for item in values)
    raise TypeError(f"{name} must be a real scalar or a sequence of them")


def _validate_positive_dim(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_positive_sigma(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def coverage_lattice(
    bounds_in: tuple[float | Sequence[float], float | Sequence[float]],
    bounds_out: tuple[float | Sequence[float], float | Sequence[float]],
    d_in: int,
    d_out: int,
    sigma_in: float,
    sigma_out: float,
    *,
    K: int | None = None,
    spacing: float | None = None,
    c: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Axis-aligned product lattice over the joint atom coordinate space ``z=(s,t)``.

    Returns ``(s, t)`` with ``s: [M, d_in]``, ``t: [M, d_out]`` (``float64``),
    where ``M`` is the product of the per-axis point counts taken over *all*
    ``d_in + d_out`` axes jointly -- e.g. with ``d_in=d_out=1`` this is a 2D
    lattice over ``z``, and halving both ``sigma_in``/``sigma_out`` (holding
    the same ``spacing``/``c`` multiplier) roughly quadruples the atom count
    (each of the two axes roughly doubles its point count).

    ``bounds_in``/``bounds_out`` are ``(lo, hi)`` pairs mirroring
    :class:`torchcst.representation.Box`: each of ``lo``/``hi`` may be one
    scalar (applied to every axis of that side) or a sequence of length
    ``d_in``/``d_out``.

    Per-axis spacing is ``Delta_a = mult * sigma_in`` for input axes and
    ``mult * sigma_out`` for output axes, where ``mult`` is either supplied
    directly via ``spacing`` (a one-shot override of the ``c`` multiplier --
    ``spacing`` and ``c`` are the same knob under two calling conventions)
    or searched for via ``K`` (see below). Exactly one of ``K``/``spacing``
    must be given; ``c`` is otherwise unused (it exists only as the
    ``K``-search's fallback multiplier and its default value has no effect
    when ``spacing`` is given directly).

    When ``K`` is given: since the total lattice count ``M(mult)`` is
    monotonically non-increasing in ``mult`` (a larger multiplier means
    coarser spacing), a closed-form initial guess

        mult0 = (prod_a(ext_a / sigma_a) / K) ** (1 / n_axes)

    seeds a bounded, deterministic bisection on ``log(mult)`` that tracks
    the best (closest-to-``K``) integer lattice count seen across ~60
    iterations. This is a deterministic best-approximation, not an exact
    solve (the true objective is a step function of ``mult``): the search
    converges to the crossing boundary of that step function to floating
    precision, which is as exact as "best integer approximation" can mean
    here.

    No randomness is used anywhere in this function.
    """
    d_in = _validate_positive_dim(d_in, "d_in")
    d_out = _validate_positive_dim(d_out, "d_out")
    sigma_in = _validate_positive_sigma(sigma_in, "sigma_in")
    sigma_out = _validate_positive_sigma(sigma_out, "sigma_out")
    if not (isinstance(bounds_in, tuple) and len(bounds_in) == 2):
        raise TypeError("bounds_in must be a (lo, hi) tuple")
    if not (isinstance(bounds_out, tuple) and len(bounds_out) == 2):
        raise TypeError("bounds_out must be a (lo, hi) tuple")
    lo_in = _as_axis_tuple(bounds_in[0], d_in, "bounds_in[0]")
    hi_in = _as_axis_tuple(bounds_in[1], d_in, "bounds_in[1]")
    lo_out = _as_axis_tuple(bounds_out[0], d_out, "bounds_out[0]")
    hi_out = _as_axis_tuple(bounds_out[1], d_out, "bounds_out[1]")
    for name, lows, highs in (("bounds_in", lo_in, hi_in), ("bounds_out", lo_out, hi_out)):
        for lo, hi in zip(lows, highs):
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"{name} must be finite")
            if lo >= hi:
                raise ValueError(f"{name} lo must be less than hi on every axis")

    if K is not None and spacing is not None:
        raise ValueError("exactly one of K or spacing must be given, not both")
    if K is None and spacing is None:
        raise ValueError("exactly one of K or spacing must be given")
    if spacing is not None:
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
            raise TypeError("spacing must be a real number")
        spacing = float(spacing)
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
    if K is not None:
        if isinstance(K, bool) or not isinstance(K, int):
            raise TypeError("K must be an int")
        if K <= 0:
            raise ValueError("K must be positive")
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise TypeError("c must be a real number")
    c = float(c)
    if not math.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")

    lo_per_axis = lo_in + lo_out
    hi_per_axis = hi_in + hi_out
    ext_per_axis = tuple(hi - lo for lo, hi in zip(lo_per_axis, hi_per_axis))
    sigma_per_axis = (sigma_in,) * d_in + (sigma_out,) * d_out
    n_axes = d_in + d_out

    def axis_counts_for(mult: float) -> tuple[int, ...]:
        counts = []
        for ext, sigma in zip(ext_per_axis, sigma_per_axis):
            delta = mult * sigma
            counts.append(max(1, int(round(ext / delta)) + 1))
        return tuple(counts)

    def total_count(mult: float) -> int:
        total = 1
        for n in axis_counts_for(mult):
            total *= n
        return total

    if spacing is not None:
        mult = spacing
    else:
        assert K is not None
        log_ratio = sum(
            math.log(ext / sigma) for ext, sigma in zip(ext_per_axis, sigma_per_axis)
        )
        mult0 = math.exp((log_ratio - math.log(K)) / n_axes)

        lo_mult = mult0 * 1e-3
        hi_mult = mult0 * 1e3
        for _ in range(20):
            if total_count(lo_mult) >= K:
                break
            lo_mult *= 0.1
        for _ in range(20):
            if total_count(hi_mult) <= K:
                break
            hi_mult *= 10.0

        lo_log, hi_log = math.log(lo_mult), math.log(hi_mult)
        best_mult, best_diff = mult0, abs(total_count(mult0) - K)
        for _ in range(60):
            mid_log = 0.5 * (lo_log + hi_log)
            mid_mult = math.exp(mid_log)
            count = total_count(mid_mult)
            diff = abs(count - K)
            if diff < best_diff:
                best_diff, best_mult = diff, mid_mult
            if count > K:
                lo_log = mid_log
            elif count < K:
                hi_log = mid_log
            else:
                best_mult = mid_mult
                break
        mult = best_mult

    counts = axis_counts_for(mult)
    axis_positions = [
        torch.linspace(lo, hi, n, dtype=torch.float64)
        for lo, hi, n in zip(lo_per_axis, hi_per_axis, counts)
    ]
    grids = torch.meshgrid(*axis_positions, indexing="ij")
    lattice = torch.stack([grid.reshape(-1) for grid in grids], dim=1)
    return lattice[:, :d_in].contiguous(), lattice[:, d_in:].contiguous()


# ---------------------------------------------------------------------------
# Function 2: logdet_greedy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreedySelection:
    """Result of one :func:`logdet_greedy` run, in selection order.

    ``gains[i]`` is the log marginal gain accepted at step ``i`` --
    ``log(Gamma_zz + ridge - Gamma_zS (Gamma_SS + ridge*I)^{-1} Gamma_Sz)``
    for the candidate selected at that step, evaluated against the set
    selected in the previous steps.
    """

    indices: Tensor  # int64 [k_selected], positions into the candidate pool
    s: Tensor  # [k_selected, d_in]
    t: Tensor  # [k_selected, d_out]
    gains: Tensor  # float [k_selected]


def _validate_real(value: float, name: str, *, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def logdet_greedy(
    candidates_s: Tensor,
    candidates_t: Tensor,
    *,
    U: Tensor | None = None,
    V: Tensor | None = None,
    U_fn: Callable[[Tensor], Tensor] | None = None,
    V_fn: Callable[[Tensor], Tensor] | None = None,
    K: int | None = None,
    rent: float | None = None,
    ridge: float,
    data: Tensor | None = None,
    jitter: float = 1e-8,
) -> GreedySelection:
    """Greedy D-optimal (DPP-MAP-style) candidate selection via incremental Cholesky.

    At each step, adds the candidate ``z`` maximizing the marginal log-det
    gain

        delta_J(z; S) = log(Gamma_zz + ridge - Gamma_zS (Gamma_SS + ridge*I)^-1 Gamma_Sz)

    ``ridge`` regularizes the whole matrix ``Gamma_D + ridge*I`` on its
    diagonal, mirroring exactly how
    :meth:`torchcst.representation.gram.GramService._assess` ridges
    ``Gamma_SS`` before solving -- the design doc's formula (section 6/2)
    does not spell out where ``ridge`` enters; this is this module's
    concrete, GramService-consistent choice. The quantity above is exactly
    what a **pivoted (incomplete) Cholesky factorization** of
    ``Gamma_D + ridge*I`` computes as its running diagonal after each pivot,
    so that is how this is implemented: one rank-1 diagonal update per
    accepted candidate, no step ever re-factorizes anything from scratch,
    and no ``[M, M]`` Gram is ever materialized (only the diagonal, computed
    once, and one column per accepted pivot).

    Stops after ``K`` selections, or as soon as the best remaining marginal
    gain drops below ``log(rent)`` (checked *before* accepting that
    candidate -- ``rent`` is the K-dial: candidates cheaper than renting one
    more atom are never bought), or when every remaining candidate's
    residual variance has collapsed to at or below ``jitter`` (the
    eps-null guard: near-duplicate or data-invisible directions must not be
    bought, mirroring the design's guard on ``GramService``). At least one
    of ``K``/``rent`` must be given.

    Exactly one of ``(U, V)`` (precomputed candidate factor matrices,
    ``U: [n_out, M]``, ``V: [n_in, M]``) or ``(U_fn, V_fn)`` (called once as
    ``U_fn(candidates_t)`` / ``V_fn(candidates_s)``) must be supplied.
    """
    if not isinstance(candidates_s, Tensor) or not isinstance(candidates_t, Tensor):
        raise TypeError("candidates_s and candidates_t must be Tensors")
    if candidates_s.ndim != 2 or candidates_t.ndim != 2:
        raise ValueError("candidates_s and candidates_t must be rank 2")
    if candidates_s.shape[0] != candidates_t.shape[0]:
        raise ValueError("candidates_s and candidates_t must share their row count")

    have_uv = U is not None or V is not None
    have_fn = U_fn is not None or V_fn is not None
    if have_uv and have_fn:
        raise ValueError("supply either (U, V) or (U_fn, V_fn), not both")
    if have_uv:
        if U is None or V is None:
            raise ValueError("both U and V must be given together")
    elif have_fn:
        if U_fn is None or V_fn is None:
            raise ValueError("both U_fn and V_fn must be given together")
        U = U_fn(candidates_t)
        V = V_fn(candidates_s)
    else:
        raise ValueError("supply either (U, V) or (U_fn, V_fn)")

    if not isinstance(U, Tensor) or not isinstance(V, Tensor):
        raise TypeError("U and V must be Tensors")
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be rank 2")
    m_candidates = candidates_s.shape[0]
    if U.shape[1] != m_candidates or V.shape[1] != m_candidates:
        raise ValueError("U and V must have one column per candidate")

    if K is None and rent is None:
        raise ValueError("at least one of K or rent must be given")
    if K is not None:
        if isinstance(K, bool) or not isinstance(K, int):
            raise TypeError("K must be an int")
        if K <= 0:
            raise ValueError("K must be positive")
        K = min(K, m_candidates)
    rent = _validate_real(rent, "rent", allow_none=True)
    if rent is not None and rent <= 0:
        raise ValueError("rent must be positive")
    ridge = _validate_real(ridge, "ridge")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    jitter = _validate_real(jitter, "jitter")
    if jitter <= 0:
        raise ValueError("jitter must be positive")

    sigma: Tensor | None = None
    if data is not None:
        if not isinstance(data, Tensor):
            raise TypeError("data must be a Tensor or None")
        if data.ndim != 2 or data.shape[1] != V.shape[0]:
            raise ValueError("data must have shape [n, n_in] matching V's rows")
        data = data.detach().to(device=V.device, dtype=V.dtype)
        sigma = (data.transpose(0, 1) @ data) / data.shape[0]

    dtype = U.dtype
    device = U.device
    diag = (U * U).sum(dim=0)
    v_metric = V if sigma is None else sigma @ V
    diag = diag * (V * v_metric).sum(dim=0) + ridge

    alive = torch.ones(m_candidates, dtype=torch.bool, device=device)
    l_cols: list[Tensor] = []
    indices: list[int] = []
    gains: list[float] = []
    log_rent = math.log(rent) if rent is not None else None

    while True:
        if K is not None and len(indices) >= K:
            break
        masked_diag = torch.where(
            alive, diag, torch.full_like(diag, float("-inf"))
        )
        if not bool(alive.any()):
            break
        best_value, z_star = torch.max(masked_diag, dim=0)
        best_value = float(best_value)
        z_star = int(z_star)
        if not math.isfinite(best_value) or best_value <= jitter:
            break
        log_gain = math.log(best_value)
        if log_rent is not None and log_gain < log_rent:
            break

        indices.append(z_star)
        gains.append(log_gain)
        l_pp = math.sqrt(best_value)
        u_p = U[:, z_star]
        v_p = V[:, z_star]
        v_p_metric = v_p if sigma is None else sigma @ v_p
        raw_col = (U.transpose(0, 1) @ u_p) * (V.transpose(0, 1) @ v_p_metric)
        if l_cols:
            l_matrix = torch.stack(l_cols, dim=1)  # [M, k]
            proj = l_matrix @ l_matrix[z_star, :]
        else:
            proj = torch.zeros(m_candidates, dtype=dtype, device=device)
        new_col = (raw_col - proj) / l_pp
        new_col[z_star] = l_pp
        l_cols.append(new_col)
        diag = diag - new_col * new_col
        alive[z_star] = False

    idx_tensor = torch.tensor(indices, dtype=torch.int64, device=device)
    gains_tensor = torch.tensor(gains, dtype=torch.float64, device=device)
    return GreedySelection(
        indices=idx_tensor,
        s=candidates_s.index_select(0, idx_tensor),
        t=candidates_t.index_select(0, idx_tensor),
        gains=gains_tensor,
    )


# ---------------------------------------------------------------------------
# Function 3: variance_amplitudes
# ---------------------------------------------------------------------------


def variance_amplitudes(
    *,
    U: Tensor | None = None,
    V: Tensor | None = None,
    Gamma: Tensor | None = None,
    data: Tensor | None = None,
    target_second_moment: float,
    generator: torch.Generator,
) -> Tensor:
    """Random +/-kappa amplitudes matching a target second moment.

    Draws a random sign vector ``sign in {-1, +1}^K`` from ``generator``,
    then solves ``kappa`` in closed form so that
    ``w = kappa * sign`` satisfies ``E||Wx||^2 = w^T Gamma_D w ==
    target_second_moment`` exactly:

        kappa = sqrt(target_second_moment / (sign^T Gamma_D sign))

    Exactly one of ``Gamma`` (a precomputed ``[K, K]`` D-weighted Gram) or
    ``(U, V)`` (factor matrices the Gram is assembled from, via the same
    Hadamard convention as :class:`torchcst.representation.gram.GramService`,
    optionally D-weighted by ``data``) must be given. Unlike
    :func:`logdet_greedy`, a dense ``[K, K]`` Gram is legitimate here: this
    runs once over the small, already-selected atom set, not a candidate
    pool or a hot-path live population.
    """
    have_gamma = Gamma is not None
    have_uv = U is not None or V is not None
    if have_gamma and have_uv:
        raise ValueError("supply either Gamma or (U, V), not both")
    if have_gamma:
        if data is not None:
            raise ValueError("data is only used with the (U, V) path")
        if not isinstance(Gamma, Tensor):
            raise TypeError("Gamma must be a Tensor")
        if Gamma.ndim != 2 or Gamma.shape[0] != Gamma.shape[1]:
            raise ValueError("Gamma must be a square matrix")
    elif have_uv:
        if U is None or V is None:
            raise ValueError("both U and V must be given together")
        if not isinstance(U, Tensor) or not isinstance(V, Tensor):
            raise TypeError("U and V must be Tensors")
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("U and V must be rank 2")
        if U.shape[1] != V.shape[1]:
            raise ValueError("U and V must have the same number of columns")
        sigma: Tensor | None = None
        if data is not None:
            if not isinstance(data, Tensor):
                raise TypeError("data must be a Tensor or None")
            if data.ndim != 2 or data.shape[1] != V.shape[0]:
                raise ValueError("data must have shape [n, n_in] matching V's rows")
            data = data.detach().to(device=V.device, dtype=V.dtype)
            sigma = (data.transpose(0, 1) @ data) / data.shape[0]
        ut_u = U.transpose(0, 1) @ U
        v_metric = V if sigma is None else sigma @ V
        vt_dv = V.transpose(0, 1) @ v_metric
        Gamma = ut_u * vt_dv
    else:
        raise ValueError("supply either Gamma or (U, V)")

    if isinstance(target_second_moment, bool) or not isinstance(
        target_second_moment, (int, float)
    ):
        raise TypeError("target_second_moment must be a real number")
    target_second_moment = float(target_second_moment)
    if not math.isfinite(target_second_moment) or target_second_moment <= 0:
        raise ValueError("target_second_moment must be finite and positive")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    k_atoms = Gamma.shape[0]
    sign_int = torch.randint(0, 2, (k_atoms,), generator=generator) * 2 - 1
    sign = sign_int.to(device=Gamma.device, dtype=Gamma.dtype)
    denom = float(sign @ (Gamma @ sign))
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError(
            "sign^T Gamma sign must be finite and positive to solve for kappa "
            "(a non-positive value means Gamma is degenerate/non-PSD along "
            "this sign direction)"
        )
    kappa = math.sqrt(target_second_moment / denom)
    return kappa * sign


# ---------------------------------------------------------------------------
# Helper: wrap into list[SynapseBirth]
# ---------------------------------------------------------------------------


def to_synapse_births(
    site: str, s: Tensor, t: Tensor, w: Tensor, *, lineage_start: int = 0
) -> list[SynapseBirth]:
    """Wrap a placed ``(s, t, w)`` batch into a single-op ``list[SynapseBirth]``.

    Mirrors ``examples/gaussian_gradient_birth.py``'s
    ``synapses.apply([SynapseBirth(...)])`` usage: one birth op carries the
    whole batch via stacked tensors, which is the shape
    ``SynapseStore.prepare``/``apply`` expects.
    """
    if not isinstance(site, str) or not site:
        raise ValueError("site must be a non-empty string")
    if not isinstance(s, Tensor) or not isinstance(t, Tensor) or not isinstance(w, Tensor):
        raise TypeError("s, t, and w must be Tensors")
    if s.ndim != 2 or t.ndim != 2 or w.ndim != 1:
        raise ValueError("s and t must be rank 2 and w must be rank 1")
    n = w.shape[0]
    if s.shape[0] != n or t.shape[0] != n:
        raise ValueError("s, t, and w must share their atom count")
    if isinstance(lineage_start, bool) or not isinstance(lineage_start, int):
        raise TypeError("lineage_start must be an int")
    lineage = torch.arange(lineage_start, lineage_start + n, dtype=torch.int64)
    return [SynapseBirth(site, s, t, w, lineage)]
