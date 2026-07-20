"""Calibration helpers that turn newborn rise time into immunity events."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from inspect import Signature, signature
from math import ceil, isfinite
from typing import Any, NamedTuple

import torch
from torch import Tensor

from cstf.audit import AuditRecord, AuditSubscriber


class TauRiseResult(NamedTuple):
    """Observed event count and whether it is only a right-censored bound."""

    tau: int
    censored: bool

    @property
    def events(self) -> int:
        return self.tau


class ImmunityCalibration(NamedTuple):
    """Pinned immunity constant and the calibration sample's censoring rate."""

    immunity_events: int
    censored_rate: float


def _trajectory_points(
    trajectory: Sequence[float] | Tensor | Mapping[int, float],
) -> tuple[list[int], list[float]]:
    if isinstance(trajectory, Tensor):
        if trajectory.ndim != 1:
            raise ValueError("trajectory Tensor must be rank 1")
        raw = trajectory.detach().to(device="cpu", dtype=torch.float64).tolist()
        events = list(range(len(raw)))
    elif isinstance(trajectory, Mapping):
        if not trajectory:
            return [], []
        if not all(
            not isinstance(event, bool) and isinstance(event, int)
            for event in trajectory
        ):
            raise TypeError("trajectory event keys must be ints")
        events = sorted(trajectory)
        raw = [trajectory[event] for event in events]
    else:
        raw = list(trajectory)
        events = list(range(len(raw)))
    values: list[float] = []
    for value in raw:
        reading = float(value)
        if not isfinite(reading) or reading < 0:
            raise ValueError("mass trajectory values must be finite and non-negative")
        values.append(reading)
    return events, values


def tau_rise(
    trajectory: Sequence[float] | Tensor | Mapping[int, float],
    steady_fraction: float = 0.9,
) -> TauRiseResult:
    """Measure events from birth to first arrival at a terminal steady level.

    The steady value is the trailing moving-average reading, using up to the
    final three observations.  Event zero (or the first mapping key) is birth.
    An empty observation column is a right-censored zero-event lower bound.
    """
    fraction = float(steady_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("steady_fraction must be in (0, 1]")
    events, values = _trajectory_points(trajectory)
    if not values:
        return TauRiseResult(0, True)
    window = min(3, len(values))
    steady = sum(values[-window:]) / window
    threshold = steady * fraction
    birth_event = events[0]
    for event, value in zip(events, values):
        if value >= threshold:
            return TauRiseResult(event - birth_event, False)
    return TauRiseResult(events[-1] - birth_event, True)


def immunity_from_tau(
    taus: Sequence[int | float | TauRiseResult], *, quantile: float = 1.0
) -> ImmunityCalibration:
    """Convert rise observations to a conservative immunity event constant.

    With the default quantile this is the Phase 3 §8B birth-step-crossing max.
    One additional event is retained as the frozen conversion-rule safety
    margin. Censored observations contribute their observed lower bound and are
    reported separately rather than silently discarded.
    """
    q = float(quantile)
    if not 0.0 < q <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    values: list[float] = []
    censored = 0
    for item in taus:
        if isinstance(item, TauRiseResult):
            value = item.tau
            censored += int(item.censored)
        else:
            value = item
        reading = float(value)
        if not isfinite(reading) or reading < 0:
            raise ValueError("tau values must be finite and non-negative")
        values.append(reading)
    if not values:
        raise ValueError("taus must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    quantile_value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return ImmunityCalibration(ceil(quantile_value) + 1, censored / len(values))


class CalibrationRun(AuditSubscriber):
    """Run a scripted train callback and collect per-newborn mass trajectories.

    The callback accepts ``(engine, one_based_update_step)`` or just
    ``(engine)`` and owns the numerical training update.  This helper then
    calls ``engine.step()``. It observes only engine-pushed audit records and
    never influences policy or structural decisions.
    """

    def __init__(self, engine: Any, train_step: Callable[..., None]) -> None:
        from cstf.engine import StructuralEngine

        if not isinstance(engine, StructuralEngine):
            raise TypeError("engine must be a StructuralEngine")
        if not callable(train_step):
            raise TypeError("train_step must be callable")
        self.engine = engine
        self.train_step = train_step
        callback_signature: Signature = signature(train_step)
        try:
            callback_signature.bind(engine, 1)
        except TypeError:
            try:
                callback_signature.bind(engine)
            except TypeError as exc:
                raise TypeError(
                    "train_step must accept (engine, update_step) or (engine)"
                ) from exc
            self._callback_takes_step = False
        else:
            self._callback_takes_step = True
        self._known_ids = {
            site: set(store.view().ids.tolist())
            for site, store in engine.stores.items()
        }
        self._trajectories: dict[tuple[str, int], list[float]] = {}
        self._events: dict[tuple[str, int], list[int]] = {}
        engine.subscribe_audit(self)

    def push(self, record: AuditRecord) -> None:
        assert record.live_ids is not None
        for site, ids in record.live_ids.items():
            masses = record.mass_snapshots[site]
            current = {int(entity_id) for entity_id in ids.tolist()}
            known = self._known_ids.setdefault(site, set())
            for position, raw_id in enumerate(ids.tolist()):
                entity_id = int(raw_id)
                key = (site, entity_id)
                if entity_id not in known:
                    self._trajectories[key] = []
                    self._events[key] = []
                if key in self._trajectories:
                    self._trajectories[key].append(float(masses[position]))
                    self._events[key].append(record.event_index)
            known.update(current)

    @property
    def trajectories(self) -> dict[tuple[str, int], tuple[float, ...]]:
        return {key: tuple(values) for key, values in self._trajectories.items()}

    @property
    def event_trajectories(self) -> dict[tuple[str, int], dict[int, float]]:
        return {
            key: dict(zip(self._events[key], values))
            for key, values in self._trajectories.items()
        }

    def run(self, updates: int) -> dict[tuple[str, int], tuple[float, ...]]:
        if isinstance(updates, bool) or not isinstance(updates, int):
            raise TypeError("updates must be an int")
        if updates < 0:
            raise ValueError("updates must be non-negative")
        for _ in range(updates):
            update_step = self.engine.clock.update_step + 1
            if self._callback_takes_step:
                self.train_step(self.engine, update_step)
            else:
                self.train_step(self.engine)
            self.engine.step()
        return self.trajectories
