"""構造更新の時刻と量だけを決めるSchedule。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


@dataclass(frozen=True)
class Clock:
    step: int


@dataclass(frozen=True)
class UpdateRequest:
    step: int
    fraction: float
    rng: torch.Generator


class UpdateSchedule(Protocol):
    def poll(
        self, clock: Clock, rng: torch.Generator
    ) -> UpdateRequest | None: ...


class PeriodicSchedule:
    """一定間隔で発火し、更新率は任意callableから得る。"""

    def __init__(
        self,
        every: int = 500,
        until: int = 50_000,
        fraction: Callable[[int], float] = lambda step: 0.3 * (
            1 - step / 50_000
        ),
    ):
        if every <= 0:
            raise ValueError("every must be positive")
        if until < 0:
            raise ValueError("until must be non-negative")
        self.every = every
        self.until = until
        self.fraction = fraction

    def poll(
        self, clock: Clock, rng: torch.Generator
    ) -> UpdateRequest | None:
        if clock.step >= self.until or clock.step % self.every:
            return None
        fraction = float(self.fraction(clock.step))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"schedule fraction must be in [0, 1], got {fraction}"
            )
        return UpdateRequest(clock.step, fraction, rng)
