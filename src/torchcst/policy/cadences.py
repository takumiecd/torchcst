"""Budget-free timing policies for structural observation and events."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import Clock, EventSignal, Phase


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PeriodicCadence:
    """Fire at a fixed update interval without deciding operation supply."""

    event_interval: int = 500
    freeze_event: int | None = None
    observe_window: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.event_interval, "event_interval")
        if self.freeze_event is not None:
            _positive_int(self.freeze_event, "freeze_event")
        if isinstance(self.observe_window, bool) or not isinstance(
            self.observe_window, int
        ):
            raise TypeError("observe_window must be an int")
        if not 0 <= self.observe_window <= self.event_interval:
            raise ValueError("observe_window must be in [0, event_interval]")

    def phase(self, clock: Clock) -> Phase:
        if self.freeze_event is not None and clock.event_index >= self.freeze_event:
            return Phase.FROZEN
        return Phase.GROW

    def event(self, clock: Clock) -> EventSignal | None:
        if clock.update_step == 0 or clock.update_step % self.event_interval:
            return None
        return EventSignal(clock.event_index, self.phase(clock))

    def observing(self, clock: Clock) -> bool:
        if clock.update_step == 0 or self.observe_window == 0:
            return False
        if self.phase(clock) is Phase.FROZEN:
            return False
        distance_to_event = (-clock.update_step) % self.event_interval
        return distance_to_event < self.observe_window


@dataclass(frozen=True)
class BirthWindowCadence:
    """Grow, sweep, freeze, and optional response timing without quotas."""

    event_interval: int = 200
    birth_end_event: int = 5
    freeze_event: int | None = 10
    observe_window: int = 0
    response_events: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _positive_int(self.event_interval, "event_interval")
        _positive_int(self.birth_end_event, "birth_end_event")
        if self.freeze_event is not None and self.freeze_event <= self.birth_end_event:
            raise ValueError("freeze_event must follow the birth window")
        if isinstance(self.observe_window, bool) or not isinstance(
            self.observe_window, int
        ):
            raise TypeError("observe_window must be an int")
        if not 0 <= self.observe_window <= self.event_interval:
            raise ValueError("observe_window must be in [0, event_interval]")
        if self.response_events is not None:
            if (
                not isinstance(self.response_events, tuple)
                or len(self.response_events) != 2
            ):
                raise TypeError("response_events must be a (start, end) tuple")
            start, end = self.response_events
            _positive_int(start, "response start_event")
            if end < start:
                raise ValueError("response end_event must not precede start_event")
            if start <= self.birth_end_event:
                raise ValueError("response window must follow the birth window")

    def _in_response(self, event_index: int) -> bool:
        if self.response_events is None:
            return False
        return self.response_events[0] <= event_index <= self.response_events[1]

    def phase(self, clock: Clock) -> Phase:
        if self._in_response(clock.event_index):
            return Phase.RESPONSE
        if (
            self.response_events is not None
            and clock.event_index > self.response_events[1]
        ):
            return Phase.SWEEP
        if clock.event_index <= self.birth_end_event:
            return Phase.GROW
        if self.freeze_event is None or clock.event_index < self.freeze_event:
            return Phase.SWEEP
        return Phase.FROZEN

    def event(self, clock: Clock) -> EventSignal | None:
        if clock.update_step == 0 or clock.update_step % self.event_interval:
            return None
        return EventSignal(clock.event_index, self.phase(clock))

    def observing(self, clock: Clock) -> bool:
        if clock.update_step == 0 or self.observe_window == 0:
            return False
        if self.phase(clock) is Phase.FROZEN:
            return False
        distance_to_event = (-clock.update_step) % self.event_interval
        return distance_to_event < self.observe_window
