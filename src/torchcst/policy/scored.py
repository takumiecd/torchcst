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
    """Select the largest finite candidate scores with stable tie-breaking."""

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
        order = torch.argsort(
            scores.index_select(0, finite), descending=True, stable=True
        )[:count]
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
