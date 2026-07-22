"""Reusable continuous candidate scoring instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchcst.compute import ObservationTiming
from torchcst.representation import Box
from torchcst.storage import SynapseStore, SynapseView

from .base import InstrumentBuildContext, WeightedMeasurement, weighted_sum


@dataclass(frozen=True)
class CandidateSnapshot:
    """Coordinates, ranking scores, and optional representation-owned lineages."""

    source: Tensor
    target: Tensor
    scores: Tensor
    lineages: Tensor | None = None

    def __post_init__(self) -> None:
        if self.source.ndim != 2 or self.target.ndim != 2:
            raise ValueError("candidate coordinates must be rank 2")
        if self.scores.ndim != 1:
            raise ValueError("candidate scores must be rank 1")
        count = self.scores.numel()
        if self.source.shape[0] != count or self.target.shape[0] != count:
            raise ValueError("candidate coordinates and scores must align")
        if self.lineages is not None:
            if self.lineages.ndim != 1 or self.lineages.dtype != torch.int64:
                raise TypeError("candidate lineages must be rank-1 int64")
            if self.lineages.numel() != count:
                raise ValueError("candidate lineages and scores must align")
        object.__setattr__(self, "source", self.source.detach())
        object.__setattr__(self, "target", self.target.detach())
        object.__setattr__(self, "scores", self.scores.detach())
        if self.lineages is not None:
            object.__setattr__(self, "lineages", self.lineages.detach())


@dataclass(frozen=True)
class ContinuousGradientRequest:
    """Build one Gaussian candidate-gradient instrument per requested site."""

    pool_size: int = 4096
    decay: float = 0.9
    chunk_size: int = 128
    name: str = "continuous_gradient_scores"
    timing: ObservationTiming | str = ObservationTiming.AFTER_BACKWARD

    def __post_init__(self) -> None:
        for field_name, value in (
            ("pool_size", self.pool_size),
            ("chunk_size", self.chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an int")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        try:
            timing = ObservationTiming(self.timing)
        except ValueError as exc:
            raise ValueError("timing must be backward_inline or after_backward") from exc
        object.__setattr__(self, "timing", timing)

    def build(self, context: InstrumentBuildContext) -> ContinuousGradientScores:
        return ContinuousGradientScores(
            context.store,
            context.module,
            context.rng,
            pool_size=self.pool_size,
            decay=self.decay,
            chunk_size=self.chunk_size,
            name=self.name,
        )


class ContinuousGradientScores:
    """EMA of signed zero-weight candidate gradients over continuous domains."""

    def __init__(
        self,
        store: SynapseStore,
        module: Any,
        rng: torch.Generator,
        *,
        pool_size: int,
        decay: float,
        chunk_size: int,
        name: str,
    ) -> None:
        if not isinstance(store.spec.domain_in, Box) or not isinstance(
            store.spec.domain_out, Box
        ):
            raise TypeError("continuous gradient scores require Box domains")
        scorer = getattr(module, "candidate_weight_grads", None)
        if not callable(scorer):
            raise TypeError("compute module must provide candidate_weight_grads()")
        self.name = name
        self.store = store
        self.module = module
        self.rng = rng
        self.pool_size = pool_size
        self.decay = float(decay)
        self.chunk_size = chunk_size
        self._version = -1
        self._source = store.s.detach().new_zeros((0, store.d_in))
        self._target = store.t.detach().new_zeros((0, store.d_out))
        self._scores = store.w.detach().new_zeros(0)

    def prepare(self, view: SynapseView, module: Any) -> None:
        """Resample only after structural state changes its store version."""
        if module is not self.module:
            raise ValueError("instrument was prepared with another compute module")
        if view.version == self._version:
            return
        self._source = self.store.spec.domain_in.sample(self.pool_size, self.rng).to(
            self.store.s
        )
        self._target = self.store.spec.domain_out.sample(self.pool_size, self.rng).to(
            self.store.t
        )
        self._scores = self.store.w.detach().new_zeros(self.pool_size)
        self._version = view.version

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        if module is not self.module:
            raise ValueError("instrument measured another compute module")
        gradient = module.candidate_weight_grads(
            x,
            g_out,
            self._source,
            self._target,
            chunk_size=self.chunk_size,
        )
        return {"gradient": gradient}

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Reduce continuous candidate gradients inside backward."""
        return self._measure(module, x, g_out)

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Measure continuous candidate gradients after backward."""
        return self._measure(module, x, g_out)

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: SynapseView,
    ) -> None:
        if view.version != self._version:
            raise RuntimeError("candidate pool changed before backward finalization")
        gradient = weighted_sum(measurements, "gradient")
        if gradient is None:
            return
        if gradient.ndim != 1 or gradient.numel() != self.pool_size:
            raise ValueError("candidate gradient has the wrong shape")
        sample = gradient.detach().abs().to(self._scores)
        self._scores = (
            self.decay * self._scores + (1.0 - self.decay) * sample
        ).detach()

    def candidate_snapshot(self) -> CandidateSnapshot:
        return CandidateSnapshot(
            self._source.detach().clone(),
            self._target.detach().clone(),
            self._scores.detach().clone(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "torchcst-continuous-gradient-scores-v1",
            "version": self._version,
            "source": self._source.detach().clone(),
            "target": self._target.detach().clone(),
            "scores": self._scores.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "torchcst-continuous-gradient-scores-v1":
            raise ValueError("unsupported continuous gradient score state schema")
        self._version = int(state["version"])
        self._source = state["source"].detach().clone()
        self._target = state["target"].detach().clone()
        self._scores = state["scores"].detach().clone()
