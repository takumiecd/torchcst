"""Clock-only schedule contract tests."""

from cstf.policy import (
    BirthWindowSchedule,
    Clock,
    PeriodicSchedule,
    Phase,
)


def test_birth_window_budget_phase_and_event_interval() -> None:
    schedule = BirthWindowSchedule(
        event_interval=2,
        birth_start_event=1,
        birth_end_event=2,
        birth_budget=3,
        freeze_event=4,
    )

    assert schedule.event(Clock(1, 1)) is None
    first = schedule.event(Clock(2, 1))
    last_birth = schedule.event(Clock(4, 2))
    sweep = schedule.event(Clock(6, 3))
    frozen = schedule.event(Clock(8, 4))

    assert first is not None and first.birth_budget == 3
    assert last_birth is not None and last_birth.birth_budget == 3
    assert first.phase is last_birth.phase is Phase.GROW
    assert sweep is not None and sweep.birth_budget == 0
    assert sweep.phase is Phase.SWEEP
    assert frozen is not None and frozen.birth_budget == 0
    assert frozen.phase is Phase.FROZEN
    assert schedule.observing(Clock(2, 1)) is False


def test_periodic_schedule_fires_only_at_update_interval() -> None:
    schedule = PeriodicSchedule(event_interval=3, birth_budget=7)

    assert schedule.event(Clock(0, 0)) is None
    assert schedule.event(Clock(2, 1)) is None
    event = schedule.event(Clock(3, 1))

    assert event is not None
    assert event.birth_budget == 7
    assert event.phase is Phase.GROW
    assert schedule.observing(Clock(3, 1)) is False

