"""High-level scored-birth composition for policy authors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from torchcst._validation import require_int, require_real
from torchcst.instruments import CandidateSnapshot
from torchcst.representation.gram import GramService
from torchcst.storage import SynapseBirth, SynapseView

from .contract import ObservationRequest
from .proposers import _continuous_lineages
from .registry import RetiredCandidateRegistry


# GramService's constructor requires a positive finite ``radius`` for its
# radius-restricted neighbourhood API (``neighbors``/``residual``/``cost``/
# ``plan_chain``), but ``ScoredBirth``'s entrance-side settlement below only
# ever calls the radius-independent :meth:`GramService.gram_block` directly,
# over the *entire* live span rather than a local neighbourhood -- so this
# value is inert, never read by anything the settlement path calls.
_SETTLEMENT_RADIUS = 1.0


@dataclass(frozen=True)
class TopKSelector:
    """Select the best finite candidate scores with stable tie-breaking.

    ``candidate_cone`` says which cone the candidate's *coefficient* lives in,
    and that -- not the sign of the score -- is what decides how a score is
    ranked. This follows the birth-score definition in the theory
    (``theory/sections/03_support_dynamics.tex``, def. "一般 birth score"):
    a shape-fixed scalar candidate ``c_o in R`` is selected by ``|S_o|``, and
    only a fixed non-negative ray ``c_o >= 0`` uses ``[S_o]_+``.

    * ``"signed"`` (default): rank by ``|S_o|``. The local profile gain is
      ``S_o^2 / (2||(I-P_T)d_o||^2)`` -- quadratic in the score -- so a large
      *negative* score is exactly as good a candidate as an equally large
      positive one; the amplitude fit simply buys it with ``c < 0``. Every
      synapse atom in this library is such a candidate, and this is also what
      discrete RigL does when it ranks dormant weights by ``|grad|``.
      Ranking those by the signed score silently discards the whole negative
      half of the leaderboard.
    * ``"nonnegative_ray"``: rank by ``[S_o]_+ = max(S_o, 0)``. Reserved for a
      candidate whose coefficient is pinned to a non-negative ray -- a neuron
      gate with a fixed shape. There a negative score really does mean "no
      first-order gain in the admissible direction", so every non-positive
      candidate is worth the same (zero) and they tie, broken by index.
      Note the theory's own caveat: if such a gate's family admits both
      ``d`` and ``-d`` as shapes, the candidate set is symmetric again and
      the correct rule reverts to ``"signed"``.

    Instruments differ in which convention their scores already carry:
    ``GradientField``/``ContinuousGradientField``/``ScoredCandidates`` emit
    ``|grad|`` (so ``"signed"`` is a no-op on them), while
    ``ContinuousCandidateField`` emits the signed inner product
    ``<G, a_z>`` (raw) or its deflated analogue, where it is load-bearing.
    """

    candidate_cone: str = "signed"

    def __post_init__(self) -> None:
        if self.candidate_cone not in ("signed", "nonnegative_ray"):
            raise ValueError(
                "candidate_cone must be 'signed' or 'nonnegative_ray', "
                f"got {self.candidate_cone!r}"
            )

    def rank_values(self, scores: Tensor) -> Tensor:
        """Map raw candidate scores to the quantity this cone ranks by."""
        if self.candidate_cone == "signed":
            return scores.abs()
        return scores.clamp_min(0.0)

    def select(self, scores: Tensor, budget: int) -> Tensor:
        if scores.ndim != 1:
            raise ValueError("candidate scores must be rank 1")
        require_int(budget, "budget", minimum=0)
        finite = torch.nonzero(torch.isfinite(scores), as_tuple=False).flatten()
        count = min(budget, finite.numel())
        if count == 0:
            return torch.zeros(0, dtype=torch.int64, device=scores.device)
        ranked = self.rank_values(scores.index_select(0, finite))
        order = torch.argsort(ranked, descending=True, stable=True)[:count]
        return finite.index_select(0, order)


@dataclass
class ScoredBirth:
    """Turn any candidate-score instrument into a budgeted birth proposer.

    The bound instrument is the cheap pre-ranker: its scores choose the
    ``budget``-sized shortlist (via ``selector``). ``rent``, when set, is a
    per-candidate profit floor on the *exact* local profile gain in loss
    units, settled via :class:`~torchcst.representation.gram.GramService`
    over the live view -- the entrance-side half of
    ``docs/absorb-and-gram-design.md``'s stage 3b economy (the exit side is
    :class:`~torchcst.policy.absorb.AbsorbCourt`). Settlement only filters
    and re-ranks *within* the shortlist, never widens it, and never alters
    ``selector.select``'s own signed-score semantics. For each shortlisted
    candidate,

    ``gain = <G, psi>_F^2 / (2 * ||(I - P_live) psi||_D^2)``

    where ``psi`` is the un-normalized candidate atom matrix from kernel
    columns, ``P_live`` the D-orthogonal projection onto the live atom span,
    and ``G`` the instrument's accumulated certificate in raw loss-gradient
    units (its ``raw_gradient()``). The cheap instrument score's
    ``0.5 * score**2`` (Frobenius-geometry, D-blind) is *not* generally the
    loss-unit gain; this settlement is the fix, validated to 8e-11 against a
    dense D-profile computation (``cst/scripts/diag_birth_gain_calibration.py``).
    Rent gating therefore requires the instrument to expose ``.port`` (a
    :class:`~torchcst.instruments.base.KernelPort`) and ``raw_gradient()`` --
    :class:`~torchcst.instruments.ContinuousCandidateField` does; a
    ``TypeError`` is raised at settlement time otherwise.

    Acceptance: candidates with ``gain >= rent`` *and* ``gain > 0``, in
    descending exact-gain order, never more than the shortlist. The second
    condition matters only at ``rent=0``: gain is non-negative by
    construction, so without it a freshly reset all-zero certificate
    ("nothing to explain" -- see ``ContinuousCandidateField``'s R_{t-1}=0
    convention) could buy births for free.

    ``ridge`` regularizes the settlement's least-squares solve; ``data``,
    when given, is a zero-argument ``[n, n_in]`` batch provider for the D
    metric, called once per settling ``propose()`` (the identity metric is
    used when ``None``, making ``res2_D == res2_F`` exactly). ``rent`` is a
    plain float handed to each part that needs it, never a shared mutable
    object threaded between rules.
    """

    request: ObservationRequest
    selector: Any = field(default_factory=TopKSelector)
    initial_weight: float = 0.0
    rent: float | None = None
    ridge: float = 1.0e-10
    data: Callable[[], Tensor] | None = None
    requires: tuple[ObservationRequest, ...] = field(init=False)
    _instruments: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _next_lineage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ObservationRequest):
            raise TypeError("request must implement ObservationRequest")
        if not callable(getattr(self.selector, "select", None)):
            raise TypeError("selector must provide select(scores, budget)")
        if self.rent is not None:
            if isinstance(self.rent, bool) or not isinstance(self.rent, (int, float)):
                raise TypeError("rent must be a real number or None")
            self.rent = require_real(self.rent, "rent", nonnegative=True)
        self.ridge = require_real(self.ridge, "ridge", nonnegative=True)
        if self.data is not None and not callable(self.data):
            raise TypeError("data must be a callable batch provider or None")
        self.requires = (self.request,)

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        try:
            instrument = instruments[self.request.name]
        except KeyError as exc:
            raise ValueError(
                f"ScoredBirth requires instrument {self.request.name!r}"
            ) from exc
        if not callable(getattr(instrument, "candidate_snapshot", None)):
            raise TypeError("scored-birth instrument must provide candidate_snapshot()")
        self._instruments[site] = instrument

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        del rng
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        try:
            instrument = self._instruments[view.site]
        except KeyError as exc:
            raise RuntimeError(
                f"instrument is not bound for site {view.site!r}"
            ) from exc
        snapshot = instrument.candidate_snapshot()
        if not isinstance(snapshot, CandidateSnapshot):
            raise TypeError("candidate_snapshot() must return CandidateSnapshot")
        positions = self.selector.select(snapshot.scores, budget)
        if self.rent is not None and positions.numel():
            gains = self._settlement_gains(view, instrument, snapshot, positions)
            order = torch.argsort(gains, descending=True, stable=True)
            positions = positions.index_select(0, order.to(positions.device))
            gains = gains.index_select(0, order)
            # ``gain`` is provably non-negative (a squared numerator over a
            # positive denominator), so ``gain >= rent`` alone is a
            # mathematical no-op at ``rent=0``; requiring ``gain > 0`` too
            # keeps an all-zero certificate ("nothing to explain") from ever
            # buying a birth. For ``rent > 0`` this changes nothing.
            keep = (gains >= self.rent) & (gains > 0.0)
            positions = positions[keep.to(positions.device)]
        count = positions.numel()
        if count == 0:
            return ()
        source = snapshot.source.index_select(0, positions.to(snapshot.source.device))
        target = snapshot.target.index_select(0, positions.to(snapshot.target.device))
        if snapshot.lineages is None:
            lineages = _continuous_lineages(view, count, registry, self._next_lineage)
        else:
            lineages = snapshot.lineages.index_select(0, positions.cpu())
        weights = view.w.new_full((count,), float(self.initial_weight))
        return (SynapseBirth(view.site, source, target, weights, lineages),)

    def _settlement_gains(
        self,
        view: SynapseView,
        instrument: Any,
        snapshot: CandidateSnapshot,
        positions: Tensor,
    ) -> Tensor:
        """Exact loss-unit profile gain for the shortlist, via GramService.

        See the class docstring for the formula and its provenance. This
        never widens ``positions`` (the cheap pre-ranker's shortlist) --
        it only computes each shortlisted candidate's exact gain so
        :meth:`propose` can filter by ``rent`` and re-rank by true gain
        instead of the cheap instrument score.
        """
        port = getattr(instrument, "port", None)
        raw_gradient = getattr(instrument, "raw_gradient", None)
        if port is None or not callable(raw_gradient):
            raise TypeError(
                "ScoredBirth's rent gate requires an instrument exposing "
                "'.port' (a KernelPort) and 'raw_gradient()' -- e.g. "
                f"ContinuousCandidateField; got {type(instrument).__name__!r}"
            )
        gradient = raw_gradient()
        source = snapshot.source.index_select(0, positions.to(snapshot.source.device))
        target = snapshot.target.index_select(0, positions.to(snapshot.target.device))
        k_in_cand, k_out_cand = port.columns(source, target)
        numerator = ((gradient @ k_in_cand) * k_out_cand).sum(dim=0).square()

        n_candidates = source.shape[0]
        n_live = view.w.numel()
        batch = self.data() if self.data is not None else None
        if batch is not None and not isinstance(batch, Tensor):
            raise TypeError("data provider must return a Tensor")

        # One Gram over live atoms plus zero-amplitude candidate atoms: rows
        # [0, n_live) are the live span, rows [n_live, n_live + n_candidates)
        # the shortlist.
        k_in_live, k_out_live = port.columns(view.s, view.t)
        combined_w = torch.cat([view.w, view.w.new_zeros(n_candidates)])
        combined_s = torch.cat([view.s, source], dim=0)
        combined_t = torch.cat([view.t, target], dim=0)
        combined_u = torch.cat([k_out_live, k_out_cand], dim=1)
        combined_v = torch.cat([k_in_live, k_in_cand], dim=1)
        gram = GramService(
            combined_u,
            combined_v,
            combined_w,
            combined_s,
            combined_t,
            radius=_SETTLEMENT_RADIUS,
            ridge=self.ridge,
            data=batch,
        )
        idx_cand = torch.arange(
            n_live, n_live + n_candidates, dtype=torch.int64, device=combined_w.device
        )
        gamma_self, _ = gram.gram_block(idx_cand, idx_cand)
        self_norm = torch.diagonal(gamma_self).clone()
        if n_live:
            # residual = ||(I - P_live) psi||_D^2 per candidate, via the
            # ridge-regularized least-squares projection onto the live span.
            idx_live = torch.arange(n_live, dtype=torch.int64, device=combined_w.device)
            gamma_live, _ = gram.gram_block(idx_live, idx_live)
            gamma_cross, _ = gram.gram_block(idx_live, idx_cand)
            eye = torch.eye(n_live, dtype=gamma_live.dtype, device=gamma_live.device)
            alpha = torch.linalg.solve(gamma_live + self.ridge * eye, gamma_cross)
            proj = (gamma_cross * alpha).sum(dim=0)
            quad = (alpha * (gamma_live @ alpha)).sum(dim=0)
            residual = (self_norm - 2.0 * proj + quad).clamp_min(0.0)
        else:
            residual = self_norm.clamp_min(0.0)
        denom = 2.0 * residual.clamp_min(torch.finfo(residual.dtype).tiny)
        return numerator / denom
