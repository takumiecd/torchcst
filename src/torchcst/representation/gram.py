"""Position-only Gram assembly, local ridge absorb assessment, and chain planning.

Implements Stage 2 of ``docs/absorb-and-gram-design.md``. :class:`GramService`
is a read-only representation-layer service over one snapshot of live rank-one
atoms ``psi_k = outer(U[:, k], V[:, k])`` (``U`` is the output-side factor
column, ``V`` the input-side one -- the same convention as
:meth:`torchcst.instruments.base.FactorPort.columns`). It assembles the
D-weighted (data second-moment) and Frobenius Gram separably --
``Gamma_ij = (U^T U)_ij * (V^T Sigma_x V)_ij`` (Hadamard) -- screens
neighborhoods by a radius on raw coordinates ``z = (s, t)``, and solves the
local ridge-regularized least-squares absorb of one atom into its
neighborhood. This module imports only ``torch`` and stays inside
``representation`` -- no ``policy``, no ``engine``, no ``instruments``, and it
never mutates anything it is given.

Acceptance is the single number ``cost(k) == 0.5 * w[k]**2 *
residual(k).res2_D``: the exact quadratic-loss increase from replacing atom
``k`` with its neighborhood's best (ridge-biased) approximation.
``residual(k).alpha`` is only the delivery manifest a caller wraps into a
:class:`~torchcst.storage.synapse.SynapseAbsorb` -- never an acceptance
input. Because the Gram depends only on positions (never amplitudes),
:meth:`GramService.plan_chain` simulates an entire sequential absorb chain by
bookkeeping running amplitudes over the same static Gram, with no re-forward
needed.

Per the design's out-of-scope section, no global dense ``K x K`` Gram is ever
materialized here: :meth:`GramService.gram_block` only ever assembles the
submatrix requested by its index arguments (a neighborhood, never the full
live population).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from torchcst._stats import data_second_moment
from torchcst._validation import require_int, require_real


@dataclass(frozen=True)
class AbsorbAssessment:
    """One atom's local absorb assessment: cost inputs and delivery manifest.

    ``res2_D`` and ``res2_F`` are both evaluated from the *same* ridge-solved
    ``alpha`` -- ``res2_F`` is never a second optimization, only the design's
    double-audit diagnostic (``||Delta W||_F == |c_k| * sqrt(res2_F)``
    alongside the D-weighted training-loss cost). ``receivers`` holds the
    live-view *positions* (not entity ids) the ``alpha`` entries align with.
    """

    res2_D: float
    res2_F: float
    alpha: Tensor
    receivers: Tensor


@dataclass(frozen=True)
class AbsorbPlanStep:
    """One step of a simulated absorb chain, ready to wrap into a SynapseAbsorb.

    ``dying``/``receivers`` are live-view *positions*; a caller maps them to
    entity ids (e.g. via ``view.ids``) before constructing the
    :class:`~torchcst.storage.synapse.SynapseAbsorb` op.
    """

    dying: int
    receivers: Tensor
    delta_w: Tensor
    cost: float


class GramService:
    """Assemble local Gram blocks and plan absorbs over one live-atom snapshot.

    Constructed from the factor matrices for the atoms live *right now*:
    ``U [n_out, K]``, ``V [n_in, K]`` (a caller obtains these via
    ``FactorPort.columns`` on the atom coordinates), amplitudes ``w [K]``,
    and raw coordinates ``s [K, d_in]`` / ``t [K, d_out]``. ``radius``
    screens neighborhoods on the concatenated position ``z = (s, t)`` via a
    chunked ``cdist`` (chunked over the live population so no ``[K, K]``
    distance matrix needs to be held at once for a large ``K``); ``ridge``
    regularizes every local least-squares solve. ``data`` is an optional
    ``[n, n_in]`` batch defining the D metric via ``Sigma_x = X^T X / n``;
    the identity metric is used when it is omitted (making ``res2_D ==
    res2_F`` exactly).

    All indices taken and returned by this class are positions into the live
    view supplied at construction (``U``/``V``/``w``/``s``/``t`` columns or
    rows, all aligned) -- never entity ids.
    """

    def __init__(
        self,
        U: Tensor,
        V: Tensor,
        w: Tensor,
        s: Tensor,
        t: Tensor,
        *,
        radius: float,
        ridge: float,
        data: Tensor | None = None,
        chunk_size: int = 2048,
    ) -> None:
        self._validate_factors(U, V, w, s, t)
        dtype = w.dtype
        device = w.device
        radius = require_real(radius, "radius", positive=True)
        ridge = require_real(ridge, "ridge", nonnegative=True)
        require_int(chunk_size, "chunk_size", minimum=1)

        sigma: Tensor | None = None
        if data is not None:
            sigma = data_second_moment(
                data, n_in=V.shape[0], device=device, dtype=dtype
            )

        self.U = U.detach()
        self.V = V.detach()
        self.w = w.detach()
        self.s = s.detach()
        self.t = t.detach()
        self.radius = radius
        self.ridge = ridge
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        self._sigma = sigma
        self._z = torch.cat([self.s, self.t], dim=1)

    @staticmethod
    def _validate_factors(U: Tensor, V: Tensor, w: Tensor, s: Tensor, t: Tensor) -> None:
        """Check the five live-snapshot tensors for rank, alignment, and placement."""
        for name, value in (("U", U), ("V", V), ("w", w), ("s", s), ("t", t)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("U and V must be rank 2")
        if w.ndim != 1:
            raise ValueError("w must be rank 1")
        if s.ndim != 2 or t.ndim != 2:
            raise ValueError("s and t must be rank 2")
        k_live = w.numel()
        if U.shape[1] != k_live or V.shape[1] != k_live:
            raise ValueError("U and V must have K columns matching w")
        if s.shape[0] != k_live or t.shape[0] != k_live:
            raise ValueError("s and t must have K rows matching w")
        for name, value in (("U", U), ("V", V), ("w", w), ("s", s), ("t", t)):
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating dtype")
        for name, value in (("U", U), ("V", V), ("s", s), ("t", t)):
            if value.dtype != w.dtype:
                raise TypeError(f"{name} must share w's dtype")
            if value.device != w.device:
                raise TypeError(f"{name} must share w's device")

    @property
    def k_live(self) -> int:
        """Number of atoms in this snapshot."""
        return self.w.numel()

    def neighbors(self, k: int) -> Tensor:
        """Return positions of live atoms within ``radius`` of ``z_k``, excluding ``k``.

        Computed via a ``cdist`` chunked over the live population so the
        peak memory of one call stays ``O(chunk_size)`` rather than
        materializing an ``[K, K]`` distance matrix.
        """
        k = self._validate_index(k)
        k_live = self.k_live
        query = self._z[k : k + 1]
        mask = torch.zeros(k_live, dtype=torch.bool, device=self.device)
        for start in range(0, k_live, self.chunk_size):
            stop = min(start + self.chunk_size, k_live)
            distances = torch.cdist(query, self._z[start:stop]).reshape(-1)
            mask[start:stop] = distances <= self.radius
        mask[k] = False
        return torch.nonzero(mask, as_tuple=False).flatten()

    def gram_block(self, idx_i: Tensor, idx_j: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(Gamma_D, Gamma_F)`` blocks for the requested index pairs.

        Assembled separably (``(U^T U) * (V^T Sigma_x V)`` Hadamard) and only
        for the requested submatrix -- see the module docstring for why a
        global ``[K, K]`` Gram is never materialized here.
        """
        idx_i = self._as_index_tensor(idx_i)
        idx_j = self._as_index_tensor(idx_j)
        u_i = self.U.index_select(1, idx_i)
        u_j = self.U.index_select(1, idx_j)
        ut_u = u_i.transpose(0, 1) @ u_j
        v_i = self.V.index_select(1, idx_i)
        v_j = self.V.index_select(1, idx_j)
        vt_v = v_i.transpose(0, 1) @ v_j
        vt_dv = (v_i.transpose(0, 1) @ (self._sigma @ v_j)) if self._sigma is not None else vt_v
        return ut_u * vt_dv, ut_u * vt_v

    def residual(self, k: int) -> AbsorbAssessment:
        """Solve the ridge-regularized local absorb of atom ``k`` into ``neighbors(k)``."""
        k = self._validate_index(k)
        return self._assess(k, self.neighbors(k))

    def cost(self, k: int) -> float:
        """Return ``0.5 * w[k]**2 * residual(k).res2_D``, the acceptance number."""
        k = self._validate_index(k)
        return 0.5 * float(self.w[k]) ** 2 * self.residual(k).res2_D

    def local_lambda_min(self, k: int) -> float:
        """Smallest eigenvalue of the normalized (correlation) D-Gram of ``[k] + neighbors(k)``.

        Pairwise coherence is structurally blind to collective dependency
        (many individually-moderate overlaps) and fiber alignment (distinct
        positions, dependent factor vectors); this eigenvalue is the
        group-level detector for both -- ``residual`` is the assessor and
        ``plan_chain`` the planner that act once this flags a neighborhood.
        """
        k = self._validate_index(k)
        neigh = self.neighbors(k)
        idx = torch.cat(
            [torch.tensor([k], dtype=torch.int64, device=self.device), neigh]
        )
        gamma_d, _ = self.gram_block(idx, idx)
        diag = torch.diagonal(gamma_d).clamp_min(torch.finfo(gamma_d.dtype).tiny)
        inv_sqrt = diag.rsqrt()
        normalized = gamma_d * inv_sqrt.unsqueeze(0) * inv_sqrt.unsqueeze(1)
        normalized = 0.5 * (normalized + normalized.transpose(0, 1))
        return float(torch.linalg.eigvalsh(normalized).min())

    def plan_chain(
        self,
        order_hint: Tensor | None = None,
        *,
        budget: int | None = None,
        cost_cap: float | None = None,
    ) -> list[AbsorbPlanStep]:
        """Simulate a sequential absorb chain: pick cheapest, remove, repeat.

        At each step, every currently-alive candidate in ``order_hint``
        (default: every live atom) with at least one currently-alive
        neighbor is re-assessed against the *static* Gram restricted to the
        currently-alive neighbor set; the cheapest one (by ``cost_cap``, if
        given) is absorbed, its receivers' running amplitudes are updated,
        and it is marked dead. Stops when ``budget`` steps have been planned
        or no eligible candidate remains (every live atom is either outside
        ``order_hint`` or has no live neighbor left, or every remaining
        candidate exceeds ``cost_cap``).

        Positions are never touched, so this reuses the one static Gram
        computed at construction throughout -- no re-forward needed, exactly
        the "Gram depends only on positions" simulation the design specifies
        for chain planning.
        """
        if order_hint is None:
            candidates = list(range(self.k_live))
        else:
            candidates = [
                self._validate_index(int(value))
                for value in self._as_index_tensor(order_hint).tolist()
            ]
        if budget is not None:
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise TypeError("budget must be an int or None")
            if budget < 0:
                raise ValueError("budget must be non-negative")
        if cost_cap is not None:
            if isinstance(cost_cap, bool) or not isinstance(cost_cap, (int, float)):
                raise TypeError("cost_cap must be a real number or None")
            cost_cap = float(cost_cap)
            if not isfinite(cost_cap) or cost_cap < 0:
                raise ValueError("cost_cap must be finite and non-negative")

        static_neighbors = {k: self.neighbors(k) for k in candidates}
        alive = torch.ones(self.k_live, dtype=torch.bool, device=self.device)
        running_w = self.w.clone()
        steps: list[AbsorbPlanStep] = []

        while budget is None or len(steps) < budget:
            best = self._cheapest_candidate(
                candidates, static_neighbors, alive, running_w, cost_cap
            )
            if best is None:
                break
            step_cost, dying, assessment = best
            delta_w = float(running_w[dying]) * assessment.alpha
            running_w.index_add_(0, assessment.receivers, delta_w)
            running_w[dying] = 0.0
            alive[dying] = False
            steps.append(
                AbsorbPlanStep(
                    dying=dying,
                    receivers=assessment.receivers.clone(),
                    delta_w=delta_w.clone(),
                    cost=step_cost,
                )
            )
        return steps

    def _cheapest_candidate(
        self,
        candidates: list[int],
        static_neighbors: dict[int, Tensor],
        alive: Tensor,
        running_w: Tensor,
        cost_cap: float | None,
    ) -> tuple[float, int, AbsorbAssessment] | None:
        """Re-assess every eligible candidate against still-alive neighbors
        and return the cheapest ``(cost, position, assessment)``, or ``None``
        when no candidate remains eligible (dead, neighborless, or over cap)."""
        best: tuple[float, int, AbsorbAssessment] | None = None
        for k in candidates:
            if not bool(alive[k]):
                continue
            neigh_all = static_neighbors[k]
            neigh_alive = (
                neigh_all[alive[neigh_all]] if neigh_all.numel() else neigh_all
            )
            if neigh_alive.numel() == 0:
                continue
            assessment = self._assess(k, neigh_alive)
            candidate_cost = 0.5 * float(running_w[k]) ** 2 * assessment.res2_D
            if cost_cap is not None and candidate_cost > cost_cap:
                continue
            if best is None or candidate_cost < best[0]:
                best = (candidate_cost, k, assessment)
        return best

    def _assess(self, k: int, neigh: Tensor) -> AbsorbAssessment:
        idx_k = torch.tensor([k], dtype=torch.int64, device=self.device)
        gamma_d_kk, gamma_f_kk = self.gram_block(idx_k, idx_k)
        if neigh.numel() == 0:
            return AbsorbAssessment(
                res2_D=float(gamma_d_kk.reshape(())),
                res2_F=float(gamma_f_kk.reshape(())),
                alpha=self.w.new_zeros(0),
                receivers=neigh,
            )
        gamma_d_nn, gamma_f_nn = self.gram_block(neigh, neigh)
        gamma_d_nk, gamma_f_nk = self.gram_block(neigh, idx_k)
        b_d = gamma_d_nk.reshape(-1)
        b_f = gamma_f_nk.reshape(-1)
        eye = torch.eye(neigh.numel(), dtype=gamma_d_nn.dtype, device=self.device)
        alpha = torch.linalg.solve(gamma_d_nn + self.ridge * eye, b_d)
        # The ridge biases alpha away from the unregularized minimizer; res2
        # is deliberately evaluated with the *unridged* Gram/b so the bias is
        # paid honestly in the reported residual (design section 1, guard 2).
        res2_d = float(
            gamma_d_kk.reshape(()) - 2.0 * (alpha @ b_d) + alpha @ (gamma_d_nn @ alpha)
        )
        res2_f = float(
            gamma_f_kk.reshape(()) - 2.0 * (alpha @ b_f) + alpha @ (gamma_f_nn @ alpha)
        )
        return AbsorbAssessment(
            res2_D=max(res2_d, 0.0),
            res2_F=max(res2_f, 0.0),
            alpha=alpha,
            receivers=neigh,
        )

    def _validate_index(self, k: int) -> int:
        if isinstance(k, Tensor):
            if k.numel() != 1:
                raise ValueError("k must be a single index")
            k = int(k)
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if k < 0 or k >= self.k_live:
            raise IndexError(f"k={k} is out of range for {self.k_live} live atoms")
        return k

    @staticmethod
    def _as_index_tensor(idx: Tensor) -> Tensor:
        if not isinstance(idx, Tensor):
            raise TypeError("index arguments must be Tensors")
        if idx.ndim != 1:
            raise ValueError("index arguments must be rank 1")
        if idx.dtype != torch.int64:
            raise TypeError("index arguments must have dtype int64")
        return idx
