"""Lightweight backward facts queued until an update boundary."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from torch import Tensor


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
class _PendingFact:
    site: str
    x: Tensor
    g_out: Tensor
    version: int


class BackwardContext:
    """Queue detached hook facts and attach one weight per microbatch.

    Hooks only call :meth:`queue`.  Coordinate gathers, reductions, absolute
    values, and EMA updates are deliberately deferred to the engine's
    ``finalize_backward`` boundary.
    """

    def __init__(self, update_id: int) -> None:
        if isinstance(update_id, bool) or not isinstance(update_id, int):
            raise TypeError("update_id must be an int")
        if update_id < 0:
            raise ValueError("update_id must be non-negative")
        self.update_id = update_id
        self._pending: list[_PendingFact] = []
        self._observations: list[Observation] = []
        self._closed = False

    @property
    def queued(self) -> int:
        return len(self._pending) + len(self._observations)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def queue(self, site: str, x: Tensor, g_out: Tensor, version: int) -> None:
        """Append only detached raw facts; this method performs no reduction."""
        if self._closed:
            raise RuntimeError("backward context is already finalized")
        if not isinstance(x, Tensor) or not isinstance(g_out, Tensor):
            raise TypeError("x and g_out must be Tensors")
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

    def finalize(self) -> tuple[Observation, ...]:
        """Release and return the weighted queue exactly once."""
        if self._closed:
            raise RuntimeError("backward context is already finalized")
        if self._pending:
            raise RuntimeError(
                "backward facts remain without observe_microbatch(weight)"
            )
        observations, self._observations = tuple(self._observations), []
        self._closed = True
        return observations

    def clear(self) -> None:
        """Drop all references and close the context after an aborted update."""
        self._pending.clear()
        self._observations.clear()
        self._closed = True

    def __len__(self) -> int:
        return self.queued
