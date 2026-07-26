"""High-level scored-birth composition for policy authors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from torchcst.instruments import CandidateSnapshot
from torchcst.storage import SynapseBirth, SynapseView

from .contract import ObservationRequest
from .registry import RetiredCandidateRegistry


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
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an int")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        finite = torch.nonzero(torch.isfinite(scores), as_tuple=False).flatten()
        count = min(budget, finite.numel())
        if count == 0:
            return torch.zeros(0, dtype=torch.int64, device=scores.device)
        ranked = self.rank_values(scores.index_select(0, finite))
        order = torch.argsort(ranked, descending=True, stable=True)[:count]
        return finite.index_select(0, order)


@dataclass
class ScoredBirth:
    """Turn any candidate-score instrument into a budgeted birth proposer."""

    request: ObservationRequest
    selector: Any = field(default_factory=TopKSelector)
    initial_weight: float = 0.0
    requires: tuple[ObservationRequest, ...] = field(init=False)
    _instruments: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _next_lineage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ObservationRequest):
            raise TypeError("request must implement ObservationRequest")
        if not callable(getattr(self.selector, "select", None)):
            raise TypeError("selector must provide select(scores, budget)")
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

    def _fresh_lineages(
        self,
        view: SynapseView,
        count: int,
        registry: RetiredCandidateRegistry,
    ) -> Tensor:
        existing = () if view.lineages is None else view.lineages
        if isinstance(existing, Tensor):
            existing = existing.detach().cpu().tolist()
        retired = [key for site, key in registry.snapshot() if site == view.site]
        floor = max((*[int(value) for value in existing], *retired), default=-1) + 1
        start = max(self._next_lineage.get(view.site, 0), floor)
        self._next_lineage[view.site] = start + count
        return torch.arange(start, start + count, dtype=torch.int64)

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        del rng
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an int")
        if budget < 0:
            raise ValueError("budget must be non-negative")
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
        count = positions.numel()
        if count == 0:
            return ()
        source = snapshot.source.index_select(0, positions.to(snapshot.source.device))
        target = snapshot.target.index_select(0, positions.to(snapshot.target.device))
        if snapshot.lineages is None:
            lineages = self._fresh_lineages(view, count, registry)
        else:
            lineages = snapshot.lineages.index_select(0, positions.cpu())
        weights = view.w.new_full((count,), float(self.initial_weight))
        return (SynapseBirth(view.site, source, target, weights, lineages),)
