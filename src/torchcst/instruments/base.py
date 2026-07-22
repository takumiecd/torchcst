"""Public extension contracts for backward observation instruments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from torchcst.policy.registry import RetiredCandidateRegistry
from torchcst.storage import SynapseStore, SynapseView


Measurement = Mapping[str, Tensor]


@dataclass(frozen=True)
class WeightedMeasurement:
    """One instrument measurement with its microbatch aggregation weight."""

    values: Measurement
    weight: float


@dataclass(frozen=True)
class InstrumentBuildContext:
    """Framework-owned capabilities available while building one site instrument."""

    site: str
    store: SynapseStore
    module: Any
    registry: RetiredCandidateRegistry
    rng: torch.Generator


@runtime_checkable
class CaptureInstrument(Protocol):
    """Timing-neutral state lifecycle shared by all capture instruments."""

    name: str

    def prepare(self, view: SynapseView, module: Any) -> None:
        """Reconcile state before the observed forward begins."""

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: SynapseView,
    ) -> None:
        """Commit one update's weighted measurements to instrument state."""


@runtime_checkable
class InlineCaptureInstrument(Protocol):
    """Instrument capability safe to execute inside a backward tensor hook."""

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> Measurement:
        """Return small signed sufficient statistics without retaining x/g_out."""


@runtime_checkable
class DeferredCaptureInstrument(Protocol):
    """Instrument capability executed after autograd finishes."""

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> Measurement:
        """Return signed statistics from retained boundary tensors."""


def checked_measurement(value: Measurement) -> dict[str, Tensor]:
    """Validate and detach a third-party instrument measurement."""
    if not isinstance(value, Mapping):
        raise TypeError("instrument measurement stage must return a mapping")
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("measurement names must be non-empty strings")
        if not isinstance(tensor, Tensor):
            raise TypeError("instrument measurements must be Tensors")
        result[name] = tensor.detach()
    if not result:
        raise ValueError("instrument measurement must not be empty")
    return result


def weighted_sum(
    measurements: tuple[WeightedMeasurement, ...], name: str
) -> Tensor | None:
    """Sum one signed measurement component before nonlinear aggregation."""
    result: Tensor | None = None
    for measurement in measurements:
        try:
            value = measurement.values[name]
        except KeyError as exc:
            raise RuntimeError(f"measurement omitted component {name!r}") from exc
        contribution = value * float(measurement.weight)
        result = (
            contribution if result is None else result.to(contribution) + contribution
        )
    return result
