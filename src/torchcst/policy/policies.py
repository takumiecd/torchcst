"""同梱synapse Policy: SETとRigL。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.hooks import RemovableHandle

from ..backward import LinearGradRecord
from ..engine.plan import MutationPlan, SiteBatch
from ..storage.synapse import (
    SynapseBirth,
    SynapseDeath,
    SynapseView,
)
from .base import Policy
from .binding import PolicyBinding, ReadPort
from .instruments import CandidateProbe
from .schedule import PeriodicSchedule, UpdateRequest, UpdateSchedule


def _count(k_live: int, fraction: float) -> int:
    if k_live == 0 or fraction == 0.0:
        return 0
    return min(max(1, int(fraction * k_live)), k_live)


def _smallest_weight_ids(view: SynapseView, n: int) -> torch.Tensor:
    positions = torch.topk(view.w.abs(), n, largest=False).indices
    return view.ids.index_select(0, positions.to(view.ids.device))


def _default_schedule() -> PeriodicSchedule:
    return PeriodicSchedule()


class cSET(Policy):
    """small-magnitude death + random birth。gradientは購読しない。"""

    def __init__(
        self,
        sites: Iterable[str],
        *,
        schedule: UpdateSchedule | None = None,
        domain: tuple[float, float] = (0.0, 1.0),
    ):
        self.sites = tuple(sites)
        if not self.sites or len(set(self.sites)) != len(self.sites):
            raise ValueError("sites must be non-empty and unique")
        self.schedule = schedule or _default_schedule()
        self.domain = domain
        self._ports: dict[str, ReadPort[SynapseView]] = {}

    def prepare(self, binding: PolicyBinding) -> None:
        self._ports = {
            site: binding.read(site, SynapseView) for site in self.sites
        }

    def _birth_coords(
        self,
        view: SynapseView,
        n: int,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self.domain
        s = lo + (hi - lo) * torch.rand(
            n, view.s.shape[-1], generator=rng
        )
        t = lo + (hi - lo) * torch.rand(
            n, view.t.shape[-1], generator=rng
        )
        return s.to(view.s), t.to(view.t)

    def step(self, update: UpdateRequest) -> MutationPlan:
        batches: list[SiteBatch] = []
        for site, port in self._ports.items():
            view = port.view()
            n = _count(view.ids.numel(), update.fraction)
            if n == 0:
                continue
            dying = _smallest_weight_ids(view, n)
            s, t = self._birth_coords(view, n, update.rng)
            batches.append(SiteBatch(site, (
                SynapseDeath(site, dying),
                SynapseBirth(site, s, t, view.w.new_zeros(n)),
            )))
        return MutationPlan(tuple(batches))


class cRigL(Policy):
    """small-magnitude death + off-support gradient birth。"""

    def __init__(
        self,
        sites: Iterable[str],
        *,
        schedule: UpdateSchedule | None = None,
        pool: int = 4096,
        domain: tuple[float, float] = (0.0, 1.0),
        decay: float = 0.9,
        seed: int = 0,
    ):
        self.sites = tuple(sites)
        if not self.sites or len(set(self.sites)) != len(self.sites):
            raise ValueError("sites must be non-empty and unique")
        self.schedule = schedule or _default_schedule()
        self._ports: dict[str, ReadPort[SynapseView]] = {}
        self.candidates = {
            site: CandidateProbe(pool, domain, decay, seed + index)
            for index, site in enumerate(self.sites)
        }
        self._handles: list[RemovableHandle] = []

    def prepare(self, binding: PolicyBinding) -> None:
        self.close()
        self._ports = {
            site: binding.read(site, SynapseView) for site in self.sites
        }
        for site in self.sites:
            capture = binding.capture(site, LinearGradRecord)
            port = self._ports[site]
            probe = self.candidates[site]

            def observe(
                record: LinearGradRecord,
                _port: ReadPort[SynapseView] = port,
                _probe: CandidateProbe = probe,
            ) -> None:
                _probe.observe(record, _port.view())

            self._handles.append(capture.subscribe(observe))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def step(self, update: UpdateRequest) -> MutationPlan:
        batches: list[SiteBatch] = []
        for site, port in self._ports.items():
            view = port.view()
            n = _count(view.ids.numel(), update.fraction)
            if n == 0:
                continue
            try:
                candidates = self.candidates[site].snapshot()
            except RuntimeError:
                continue
            n = min(n, candidates.scores.numel())
            if n == 0:
                continue

            dying = _smallest_weight_ids(view, n)
            selected = torch.topk(candidates.scores, n).indices
            s = candidates.s.index_select(0, selected.to(candidates.s.device))
            t = candidates.t.index_select(0, selected.to(candidates.t.device))
            batches.append(SiteBatch(site, (
                SynapseDeath(site, dying),
                SynapseBirth(site, s, t, view.w.new_zeros(n)),
            )))
        return MutationPlan(tuple(batches))
