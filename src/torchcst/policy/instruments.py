"""Policyが所有するgradient/step観測器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..backward import LinearGradRecord
from ..storage.synapse import SynapseView


@dataclass(frozen=True)
class IdScores:
    ids: Tensor
    scores: Tensor


@dataclass(frozen=True)
class CandidateScores:
    s: Tensor
    t: Tensor
    scores: Tensor


class _IdSpaceEMA:
    def __init__(self, decay: float):
        self.decay = decay
        self.version = -1
        self.ids = torch.zeros(0, dtype=torch.int64)
        self.ema = torch.zeros(0)

    def _reconcile(
        self, ids: Tensor, version: int, sample: Tensor
    ) -> None:
        if self.ids.numel() and ids.numel():
            survive = torch.isin(self.ids, ids)
            old_ids = self.ids[survive]
            old_ema = self.ema[survive.to(self.ema.device)]
        else:
            old_ids = self.ids.new_zeros(0)
            old_ema = sample.new_zeros(0)

        ema = sample.new_zeros(ids.numel())
        if old_ids.numel():
            match = old_ids.unsqueeze(1) == ids.unsqueeze(0)
            positions = match.float().argmax(dim=1)
            ema[positions.to(ema.device)] = old_ema.to(ema)

        self.ids = ids
        self.ema = ema
        self.version = version

    def update(self, ids: Tensor, version: int, sample: Tensor) -> None:
        if version != self.version:
            self._reconcile(ids, version, sample)
        self.ema = self.decay * self.ema + (1.0 - self.decay) * sample

    def snapshot(self) -> IdScores:
        return IdScores(self.ids, self.ema)


class MassEMA:
    def __init__(self, decay: float = 0.9):
        self._core = _IdSpaceEMA(decay)

    def observe(self, view: SynapseView) -> None:
        self._core.update(
            view.ids, view.version, view.w.detach().abs()
        )

    def snapshot(self) -> IdScores:
        return self._core.snapshot()


class GradEMA:
    """LinearGradRecordからlive synapseの|dL/dw|を蓄積する。"""

    def __init__(self, decay: float = 0.9):
        self._core = _IdSpaceEMA(decay)

    def observe(self, record: LinearGradRecord, view: SynapseView) -> None:
        with torch.no_grad():
            sample = record.gradients.weight_gradients(
                record.input,
                record.grad_output,
                view.s,
                view.t,
            ).abs()
        self._core.update(view.ids, view.version, sample)

    def snapshot(self) -> IdScores:
        return self._core.snapshot()


class CandidateProbe:
    """off-support候補を保持し、GradRecordごとにgradient EMAを更新する。"""

    def __init__(
        self,
        pool: int = 4096,
        domain: tuple[float, float] = (0.0, 1.0),
        decay: float = 0.9,
        seed: int = 0,
    ):
        if pool <= 0:
            raise ValueError("pool must be positive")
        self.pool = pool
        self.domain = domain
        self.decay = decay
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._version = -1
        self._s: Tensor | None = None
        self._t: Tensor | None = None
        self._scores: Tensor | None = None

    def _resample(self, view: SynapseView) -> None:
        lo, hi = self.domain
        s = lo + (hi - lo) * torch.rand(
            self.pool, view.s.shape[-1], generator=self._rng
        )
        t = lo + (hi - lo) * torch.rand(
            self.pool, view.t.shape[-1], generator=self._rng
        )
        self._s = s.to(view.s)
        self._t = t.to(view.t)
        self._scores = view.w.new_zeros(self.pool)

    def observe(self, record: LinearGradRecord, view: SynapseView) -> None:
        if record.site != view.site:
            raise ValueError(
                f"record site {record.site!r} does not match view {view.site!r}"
            )
        if self._s is None or record.version != self._version:
            self._resample(view)
            self._version = record.version
        assert self._s is not None
        assert self._t is not None
        assert self._scores is not None

        with torch.no_grad():
            sample = record.gradients.weight_gradients(
                record.input,
                record.grad_output,
                self._s,
                self._t,
            ).abs()
        self._scores = self.decay * self._scores + (1.0 - self.decay) * sample

    def snapshot(self) -> CandidateScores:
        if self._s is None or self._t is None or self._scores is None:
            raise RuntimeError("CandidateProbe has not observed a gradient yet")
        return CandidateScores(self._s, self._t, self._scores)


class GateEMA:
    def observe(self, record: LinearGradRecord) -> None:
        raise NotImplementedError("GateEMA semantics are not designed yet")


class RentCounter:
    def observe(self, record: LinearGradRecord) -> None:
        raise NotImplementedError("RentCounter semantics are not designed yet")
