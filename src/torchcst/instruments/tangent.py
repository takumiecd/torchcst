"""Signed tangent sufficient statistics for fast-construction families.

The instrument records the two matrices used by the FC-0/1 tangent model:
the loss-gradient/input cross moment and the empirical input covariance.  It
does not select candidates or mutate structure; consumers may only read the
finalized evidence at a policy-tree event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchcst.compute import ObservationTiming, flatten_capture_pair
from torchcst.storage import SynapseView

from .base import InstrumentBuildContext, WeightedMeasurement, weighted_sum

__all__ = [
    "TangentSnapshot",
    "TangentStatistics",
    "TangentStatisticsRequest",
]


@dataclass(frozen=True)
class TangentSnapshot:
    """One immutable observation-window snapshot.

    ``cross`` is ``-g_out.T @ x`` in the scaling delivered by autograd.  For
    the usual mean-reduced objective this is exactly the FC-1 statistic
    ``C = -G_h.T X / N`` because PyTorch's boundary gradient already carries
    the ``1/N`` factor.  ``covariance`` is always the explicit empirical
    moment ``X.T X / N``.
    """

    cross: Tensor
    covariance: Tensor
    weighted_batches: float
    version: int

    def __post_init__(self) -> None:
        if self.cross.ndim != 2 or self.covariance.ndim != 2:
            raise ValueError("tangent statistics must be rank-2 matrices")
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("tangent covariance must be square")
        if self.cross.shape[1] != self.covariance.shape[0]:
            raise ValueError("tangent cross moment and covariance must align")
        object.__setattr__(self, "cross", self.cross.detach())
        object.__setattr__(self, "covariance", self.covariance.detach())
        object.__setattr__(self, "weighted_batches", float(self.weighted_batches))


@dataclass(frozen=True)
class TangentStatisticsRequest:
    """Frozen request for one signed tangent-statistics instrument per site."""

    name: str = "tangent_statistics"
    timing: ObservationTiming | str = ObservationTiming.AFTER_BACKWARD

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        try:
            timing = ObservationTiming(self.timing)
        except ValueError as exc:
            raise ValueError("timing must be backward_inline or after_backward") from exc
        object.__setattr__(self, "timing", timing)

    def build(self, context: InstrumentBuildContext) -> TangentStatistics:
        return TangentStatistics(context.module, name=self.name)


class TangentStatistics:
    """Accumulate signed ``(C, Sigma_x)`` evidence until policy consumption."""

    def __init__(self, module: Any, *, name: str = "tangent_statistics") -> None:
        if module is None:
            raise TypeError("tangent statistics require a compute module")
        self.name = name
        self.module = module
        self._version = -1
        self._cross: Tensor | None = None
        self._covariance: Tensor | None = None
        self._weighted_batches = 0.0

    def prepare(self, view: SynapseView, module: Any) -> None:
        if module is not self.module:
            raise ValueError("instrument was prepared with another compute module")
        if self._version not in (-1, view.version):
            self.reset()
        self._version = view.version

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        if module is not self.module:
            raise ValueError("instrument measured another compute module")
        in_features = int(getattr(module, "in_features"))
        out_features = int(getattr(module, "out_features"))
        x_flat, g_flat = flatten_capture_pair(x, g_out, in_features, out_features)
        if x_flat.shape[0] == 0:
            raise ValueError("tangent statistics require a non-empty batch")
        count = x_flat.shape[0]
        return {
            "cross": -(g_flat.transpose(0, 1) @ x_flat),
            "covariance": x_flat.transpose(0, 1) @ x_flat / count,
            "batches": x_flat.new_ones(()),
        }

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Reduce the signed matrices inside the output tensor's hook."""
        return self._measure(module, x, g_out)

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Measure the same signed matrices after autograd has completed."""
        return self._measure(module, x, g_out)

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: SynapseView,
    ) -> None:
        if view.version != self._version:
            raise RuntimeError("synapse version changed before tangent finalization")
        cross = weighted_sum(measurements, "cross")
        covariance = weighted_sum(measurements, "covariance")
        batches = weighted_sum(measurements, "batches")
        if cross is None or covariance is None or batches is None:
            return
        if cross.ndim != 2 or covariance.ndim != 2:
            raise ValueError("tangent measurements have the wrong rank")
        if covariance.shape != (cross.shape[1], cross.shape[1]):
            raise ValueError("tangent measurement dimensions do not align")
        if self._cross is None:
            self._cross = cross.detach().clone()
            self._covariance = covariance.detach().clone()
        else:
            assert self._covariance is not None
            if self._cross.shape != cross.shape or self._covariance.shape != covariance.shape:
                raise ValueError("tangent statistic dimensions changed")
            self._cross = (self._cross.to(cross) + cross).detach()
            self._covariance = (
                self._covariance.to(covariance) + covariance
            ).detach()
        self._weighted_batches += float(batches.detach().cpu())

    def snapshot(self) -> TangentSnapshot:
        if self._cross is None or self._covariance is None:
            in_features = int(getattr(self.module, "in_features"))
            out_features = int(getattr(self.module, "out_features"))
            parameter = next(self.module.parameters(), None)
            like = parameter if parameter is not None else torch.empty(())
            cross = like.new_zeros((out_features, in_features))
            covariance = like.new_zeros((in_features, in_features))
        else:
            cross = self._cross
            covariance = self._covariance
        return TangentSnapshot(
            cross.detach().clone(),
            covariance.detach().clone(),
            self._weighted_batches,
            self._version,
        )

    def reset(self) -> None:
        """Consume the current R_(t-1)=0 observation window."""
        self._cross = None
        self._covariance = None
        self._weighted_batches = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "torchcst-tangent-statistics-v1",
            "version": self._version,
            "cross": None if self._cross is None else self._cross.detach().clone(),
            "covariance": (
                None
                if self._covariance is None
                else self._covariance.detach().clone()
            ),
            "weighted_batches": self._weighted_batches,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "torchcst-tangent-statistics-v1":
            raise ValueError("unsupported tangent-statistics state schema")
        self._version = int(state["version"])
        cross = state["cross"]
        covariance = state["covariance"]
        self._cross = None if cross is None else cross.detach().clone()
        self._covariance = (
            None if covariance is None else covariance.detach().clone()
        )
        self._weighted_batches = float(state["weighted_batches"])
