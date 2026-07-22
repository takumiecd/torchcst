"""Logical structural quotas independent of cadence and storage layout."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contract import Clock, Phase, StructuralQuota


@dataclass(frozen=True)
class ConstantQuota:
    """Return one fixed quota at every non-frozen structural event."""

    value: StructuralQuota

    def __post_init__(self) -> None:
        if not isinstance(self.value, StructuralQuota):
            raise TypeError("value must be a StructuralQuota")

    def at(self, clock: Clock, phase: Phase) -> StructuralQuota:
        del clock
        return StructuralQuota.zero() if phase is Phase.FROZEN else self.value


@dataclass(frozen=True)
class QuotaWindow:
    """Inclusive event-index interval carrying one logical quota."""

    start_event: int
    end_event: int
    quota: StructuralQuota

    def __post_init__(self) -> None:
        for name, value in (
            ("start_event", self.start_event),
            ("end_event", self.end_event),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.end_event < self.start_event:
            raise ValueError("end_event must not precede start_event")
        if not isinstance(self.quota, StructuralQuota):
            raise TypeError("quota must be a StructuralQuota")

    def contains(self, event_index: int) -> bool:
        return self.start_event <= event_index <= self.end_event


@dataclass(frozen=True)
class WindowedQuota:
    """Select the first matching event window, otherwise return zero supply."""

    windows: tuple[QuotaWindow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.windows, tuple):
            raise TypeError("windows must be a tuple")
        if not all(isinstance(window, QuotaWindow) for window in self.windows):
            raise TypeError("windows must contain QuotaWindow values")
        for index, left in enumerate(self.windows):
            for right in self.windows[index + 1 :]:
                if max(left.start_event, right.start_event) <= min(
                    left.end_event, right.end_event
                ):
                    raise ValueError("quota windows must not overlap")

    def at(self, clock: Clock, phase: Phase) -> StructuralQuota:
        if phase is Phase.FROZEN:
            return StructuralQuota.zero()
        for window in self.windows:
            if window.contains(clock.event_index):
                return window.quota
        return StructuralQuota.zero()


@dataclass(frozen=True)
class CallableQuota:
    """Adapt an annealing or resource-aware function to the quota contract."""

    function: Callable[[Clock, Phase], StructuralQuota]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable")

    def at(self, clock: Clock, phase: Phase) -> StructuralQuota:
        value = self.function(clock, phase)
        if not isinstance(value, StructuralQuota):
            raise TypeError("quota function must return StructuralQuota")
        return StructuralQuota.zero() if phase is Phase.FROZEN else value
