"""Clock-only schedules for lifecycle and periodic rewiring policies."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import Clock, EventDirective, Phase


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ResponseWindow:
    """An inclusive event-index window with separate gate and atom supply."""

    start_event: int
    end_event: int
    ungates_per_event: int = 1
    birth_budget: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.start_event, "start_event")
        if self.end_event < self.start_event:
            raise ValueError("end_event must not precede start_event")
        if isinstance(self.ungates_per_event, bool) or not isinstance(
            self.ungates_per_event, int
        ):
            raise TypeError("ungates_per_event must be an int")
        if self.ungates_per_event < 0:
            raise ValueError("ungates_per_event must be non-negative")
        if isinstance(self.birth_budget, bool) or not isinstance(
            self.birth_budget, int
        ):
            raise TypeError("birth_budget must be an int")
        if self.birth_budget < 0:
            raise ValueError("birth_budget must be non-negative")

    def contains(self, event_index: int) -> bool:
        return self.start_event <= event_index <= self.end_event


@dataclass(frozen=True)
class BirthWindowSchedule:
    """Issue bounded birth supply, then sweep, then remain structurally frozen."""

    event_interval: int = 200
    birth_start_event: int = 1
    birth_end_event: int = 5
    birth_budget: int = 5
    freeze_event: int | None = 10
    observe_window: int = 0
    response_events: tuple[int, int] | None = None
    response_ungates_per_event: int = 0
    response_birth_budget: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.event_interval, "event_interval")
        _positive_int(self.birth_start_event, "birth_start_event")
        if self.birth_end_event < self.birth_start_event:
            raise ValueError("birth_end_event must not precede birth_start_event")
        if isinstance(self.birth_budget, bool) or not isinstance(
            self.birth_budget, int
        ):
            raise TypeError("birth_budget must be an int")
        if self.birth_budget < 0:
            raise ValueError("birth_budget must be non-negative")
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
            response = ResponseWindow(
                self.response_events[0],
                self.response_events[1],
                self.response_ungates_per_event,
                self.response_birth_budget,
            )
            if response.start_event <= self.birth_end_event:
                raise ValueError("response window must follow the birth window")
        else:
            if self.response_ungates_per_event or self.response_birth_budget:
                raise ValueError("response budgets require response_events")

    @property
    def response_window(self) -> ResponseWindow | None:
        if self.response_events is None:
            return None
        return ResponseWindow(
            self.response_events[0],
            self.response_events[1],
            self.response_ungates_per_event,
            self.response_birth_budget,
        )

    def phase(self, clock: Clock) -> Phase:
        response = self.response_window
        if response is not None and response.contains(clock.event_index):
            return Phase.RESPONSE
        if response is not None and clock.event_index > response.end_event:
            # After a switch response, events remain rent-only until quiescent.
            return Phase.SWEEP
        if clock.event_index <= self.birth_end_event:
            return Phase.GROW
        if self.freeze_event is None or clock.event_index < self.freeze_event:
            return Phase.SWEEP
        return Phase.FROZEN

    def event(self, clock: Clock) -> EventDirective | None:
        if clock.update_step == 0 or clock.update_step % self.event_interval:
            return None
        phase = self.phase(clock)
        in_window = (
            phase is Phase.GROW
            and self.birth_start_event <= clock.event_index <= self.birth_end_event
        )
        response = self.response_window
        if phase is Phase.RESPONSE:
            assert response is not None
            return EventDirective(
                event_index=clock.event_index,
                birth_budget=response.birth_budget,
                phase=phase,
                ungate_budget=response.ungates_per_event,
            )
        return EventDirective(
            event_index=clock.event_index,
            birth_budget=self.birth_budget if in_window else 0,
            phase=phase,
        )

    def observing(self, clock: Clock) -> bool:
        if clock.update_step == 0 or self.observe_window == 0:
            return False
        if self.phase(clock) is Phase.FROZEN:
            return False
        distance_to_event = (-clock.update_step) % self.event_interval
        return distance_to_event < self.observe_window


@dataclass(frozen=True)
class PeriodicSchedule:
    """Fire forever at a fixed update interval, or freeze at a chosen event."""

    event_interval: int = 500
    birth_budget: int = 2**31 - 1
    freeze_event: int | None = None
    observe_window: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.event_interval, "event_interval")
        if isinstance(self.birth_budget, bool) or not isinstance(
            self.birth_budget, int
        ):
            raise TypeError("birth_budget must be an int")
        if self.birth_budget < 0:
            raise ValueError("birth_budget must be non-negative")
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

    def event(self, clock: Clock) -> EventDirective | None:
        if clock.update_step == 0 or clock.update_step % self.event_interval:
            return None
        phase = self.phase(clock)
        return EventDirective(
            event_index=clock.event_index,
            birth_budget=0 if phase is Phase.FROZEN else self.birth_budget,
            phase=phase,
        )

    def observing(self, clock: Clock) -> bool:
        if clock.update_step == 0 or self.observe_window == 0:
            return False
        if self.phase(clock) is Phase.FROZEN:
            return False
        distance_to_event = (-clock.update_step) % self.event_interval
        return distance_to_event < self.observe_window
