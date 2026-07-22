"""Lightweight backward facts queued until an update boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType

from torch import Tensor


class CaptureMode(str, Enum):
    """When backward observations are reduced to instrument statistics."""

    DEFERRED = "deferred"
    INLINE_REDUCED = "inline_reduced"


ReducedPayload = Mapping[str, Tensor]
CaptureReducer = Callable[[str, Tensor, Tensor, int], ReducedPayload]


@dataclass(frozen=True)
class Observation:
    """One detached module-boundary observation from a backward pass."""

    site: str
    x: Tensor
    g_out: Tensor
    version: int
    update_id: int
    micro_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("site must be a non-empty string")
        if not isinstance(self.x, Tensor) or not isinstance(self.g_out, Tensor):
            raise TypeError("x and g_out must be Tensors")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an int")
        if isinstance(self.update_id, bool) or not isinstance(self.update_id, int):
            raise TypeError("update_id must be an int")
        weight = float(self.micro_weight)
        if not isfinite(weight):
            raise ValueError("micro_weight must be finite")
        object.__setattr__(self, "x", self.x.detach())
        object.__setattr__(self, "g_out", self.g_out.detach())
        object.__setattr__(self, "micro_weight", weight)


@dataclass(frozen=True)
class ReducedObservation:
    """One hook-time reduction, retained without its full activation tensors."""

    site: str
    values: ReducedPayload
    version: int
    update_id: int
    micro_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("site must be a non-empty string")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        values: dict[str, Tensor] = {}
        for name, value in self.values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("reduced value names must be non-empty strings")
            if not isinstance(value, Tensor):
                raise TypeError("reduced values must be Tensors")
            values[name] = value.detach()
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an int")
        if isinstance(self.update_id, bool) or not isinstance(self.update_id, int):
            raise TypeError("update_id must be an int")
        weight = float(self.micro_weight)
        if not isfinite(weight):
            raise ValueError("micro_weight must be finite")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "micro_weight", weight)


@dataclass(frozen=True)
class CaptureBatch:
    """Finalized raw and hook-reduced observations for one optimizer update."""

    observations: tuple[Observation, ...]
    reduced: tuple[ReducedObservation, ...]


@dataclass(frozen=True)
class _PendingFact:
    site: str
    x: Tensor
    g_out: Tensor
    version: int


@dataclass(frozen=True)
class _PendingReducedFact:
    site: str
    values: ReducedPayload
    version: int


class BackwardContext:
    """Queue detached hook facts and attach one weight per microbatch.

    With no site reducer, :meth:`queue` retains raw ``x``/``g_out`` facts for
    the engine's ``finalize_backward`` boundary.  Sites with reducers compute
    their sufficient statistics inside the hook and retain only those detached
    tensors.  Absolute values and EMA updates always remain update-boundary
    operations so both modes preserve ``abs_after_sum`` semantics.
    """

    def __init__(
        self,
        update_id: int,
        reducers: Mapping[str, CaptureReducer] | None = None,
    ) -> None:
        if isinstance(update_id, bool) or not isinstance(update_id, int):
            raise TypeError("update_id must be an int")
        if update_id < 0:
            raise ValueError("update_id must be non-negative")
        self.update_id = update_id
        if reducers is None:
            reducers = {}
        if not isinstance(reducers, Mapping):
            raise TypeError("reducers must be a mapping or None")
        checked: dict[str, CaptureReducer] = {}
        for site, reducer in reducers.items():
            if not isinstance(site, str) or not site:
                raise ValueError("reducer sites must be non-empty strings")
            if not callable(reducer):
                raise TypeError("capture reducers must be callable")
            checked[site] = reducer
        self._reducers = MappingProxyType(checked)
        self._pending: list[_PendingFact] = []
        self._observations: list[Observation] = []
        self._pending_reduced: list[_PendingReducedFact] = []
        self._reduced: list[ReducedObservation] = []
        self._closed = False

    @property
    def queued(self) -> int:
        return self.raw_queued + self.reduced_queued

    @property
    def raw_queued(self) -> int:
        return len(self._pending) + len(self._observations)

    @property
    def reduced_queued(self) -> int:
        return len(self._pending_reduced) + len(self._reduced)

    @property
    def pending(self) -> int:
        return len(self._pending) + len(self._pending_reduced)

    def queue(self, site: str, x: Tensor, g_out: Tensor, version: int) -> None:
        """Append raw facts or run the site's configured hook-time reducer."""
        if self._closed:
            raise RuntimeError("backward context is already finalized")
        if not isinstance(x, Tensor) or not isinstance(g_out, Tensor):
            raise TypeError("x and g_out must be Tensors")
        reducer = self._reducers.get(site)
        if reducer is not None:
            values = reducer(site, x.detach(), g_out.detach(), int(version))
            if not isinstance(values, Mapping):
                raise TypeError("capture reducer must return a mapping")
            self._pending_reduced.append(
                _PendingReducedFact(site, dict(values), int(version))
            )
            return
        self._pending.append(
            _PendingFact(site, x.detach(), g_out.detach(), int(version))
        )

    def observe_microbatch(self, weight: float = 1.0) -> None:
        """Assign ``weight`` to every fact queued since the prior boundary."""
        if self._closed:
            raise RuntimeError("backward context is already finalized")
        value = float(weight)
        if not isfinite(value):
            raise ValueError("microbatch weight must be finite")
        pending, self._pending = self._pending, []
        pending_reduced, self._pending_reduced = self._pending_reduced, []
        self._observations.extend(
            Observation(
                fact.site,
                fact.x,
                fact.g_out,
                fact.version,
                self.update_id,
                value,
            )
            for fact in pending
        )
        self._reduced.extend(
            ReducedObservation(
                fact.site,
                fact.values,
                fact.version,
                self.update_id,
                value,
            )
            for fact in pending_reduced
        )

    def finalize_capture(self) -> CaptureBatch:
        """Release raw and reduced observations together exactly once."""
        if self._closed:
            raise RuntimeError("backward context is already finalized")
        if self._pending or self._pending_reduced:
            raise RuntimeError(
                "backward facts remain without observe_microbatch(weight)"
            )
        batch = CaptureBatch(tuple(self._observations), tuple(self._reduced))
        self._observations = []
        self._reduced = []
        self._closed = True
        return batch

    def finalize(self) -> tuple[Observation, ...]:
        """Backward-compatible raw-only finalization API."""
        batch = self.finalize_capture()
        if batch.reduced:
            raise RuntimeError(
                "reduced capture requires finalize_capture(), not finalize()"
            )
        return batch.observations

    def clear(self) -> None:
        """Drop all references and close the context after an aborted update."""
        self._pending.clear()
        self._observations.clear()
        self._pending_reduced.clear()
        self._reduced.clear()
        self._closed = True

    def __len__(self) -> int:
        return self.queued
