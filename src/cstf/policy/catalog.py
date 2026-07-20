"""Step-2 policy catalog: lifecycle champion and cSET baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .bundle import BundleComposer
from .contract import EvenBudgetAllocator, Policy
from .courts import MagnitudeCourt, RentCourt
from .proposers import (
    Bounds,
    GradFieldTopKBirth,
    IncidentOutputBirth,
    OrthogonalBirth,
    UniformEntryBirth,
)
from .schedules import BirthWindowSchedule, PeriodicSchedule


@dataclass(frozen=True)
class LC:
    """Frozen 5c lifecycle constants, individually overrideable for a run."""

    event_interval: int = 200
    birth_start_event: int = 1
    birth_end_event: int = 5
    birth_budget: int = 5
    freeze_event: int | None = 10
    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    bounds_in: Bounds | None = None
    bounds_out: Bounds | None = None
    initial_weight: float = 0.0
    composer: None = field(default=None, init=False)
    profit: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        policy = Policy(
            schedule=BirthWindowSchedule(
                event_interval=self.event_interval,
                birth_start_event=self.birth_start_event,
                birth_end_event=self.birth_end_event,
                birth_budget=self.birth_budget,
                freeze_event=self.freeze_event,
            ),
            proposers=(
                UniformEntryBirth(
                    self.bounds_in, self.bounds_out, self.initial_weight
                ),
            ),
            allocator=EvenBudgetAllocator(),
            retention=RentCourt(
                immunity_events=self.immunity_events,
                rent_ratio=self.rent_ratio,
                strikes=self.strikes,
            ),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)

    @property
    def schedule(self):
        return self._policy.schedule

    @property
    def proposers(self):
        return self._policy.proposers

    @property
    def allocator(self):
        return self._policy.allocator

    @property
    def retention(self):
        return self._policy.retention

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy


@dataclass(frozen=True)
class LC_anti:
    """5c lifecycle with B4 output-subspace-avoiding rank-one births."""

    event_interval: int = 200
    birth_start_event: int = 1
    birth_end_event: int = 5
    birth_budget: int = 5
    freeze_event: int | None = 10
    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    observe_window: int = 1
    certificate_rank: int = 1
    composer: None = field(default=None, init=False)
    profit: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        policy = Policy(
            schedule=BirthWindowSchedule(
                event_interval=self.event_interval,
                birth_start_event=self.birth_start_event,
                birth_end_event=self.birth_end_event,
                birth_budget=self.birth_budget,
                freeze_event=self.freeze_event,
                observe_window=self.observe_window,
            ),
            proposers=(OrthogonalBirth(rank=self.certificate_rank),),
            allocator=EvenBudgetAllocator(),
            retention=RentCourt(
                immunity_events=self.immunity_events,
                rent_ratio=self.rent_ratio,
                strikes=self.strikes,
            ),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)

    @property
    def schedule(self):
        return self._policy.schedule

    @property
    def proposers(self):
        return self._policy.proposers

    @property
    def allocator(self):
        return self._policy.allocator

    @property
    def retention(self):
        return self._policy.retention

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy


@dataclass(frozen=True)
class LC_response:
    """5c lifecycle plus a scheduled Phase 3-B ungate response window."""

    event_interval: int = 200
    birth_start_event: int = 1
    birth_end_event: int = 5
    birth_budget: int = 5
    freeze_event: int | None = 10
    response_events: tuple[int, int] = (11, 15)
    response_ungates_per_event: int = 1
    response_birth_budget: int = 5
    incident_births: int = 5
    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    bounds_in: Bounds | None = None
    bounds_out: Bounds | None = None
    initial_weight: float = 0.0
    initial_gate: float = 1.0e-3
    profit: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        composer = BundleComposer(
            incident_births=self.incident_births,
            initial_gate=self.initial_gate,
        )
        retention = RentCourt(
            immunity_events=self.immunity_events,
            rent_ratio=self.rent_ratio,
            strikes=self.strikes,
        )
        neuron_retention = RentCourt(
            immunity_events=self.immunity_events,
            rent_ratio=self.rent_ratio,
            strikes=self.strikes,
        )
        policy = Policy(
            schedule=BirthWindowSchedule(
                event_interval=self.event_interval,
                birth_start_event=self.birth_start_event,
                birth_end_event=self.birth_end_event,
                birth_budget=self.birth_budget,
                freeze_event=self.freeze_event,
                response_events=self.response_events,
                response_ungates_per_event=self.response_ungates_per_event,
                response_birth_budget=self.response_birth_budget,
            ),
            proposers=(
                UniformEntryBirth(
                    self.bounds_in, self.bounds_out, self.initial_weight
                ),
                IncidentOutputBirth(self.initial_weight),
            ),
            allocator=EvenBudgetAllocator(),
            retention=retention,
            composer=composer,
            profit=None,
            neuron_retention=neuron_retention,
        )
        object.__setattr__(self, "_policy", policy)

    @property
    def schedule(self):
        return self._policy.schedule

    @property
    def proposers(self):
        return self._policy.proposers

    @property
    def allocator(self):
        return self._policy.allocator

    @property
    def retention(self):
        return self._policy.retention

    @property
    def neuron_retention(self):
        return self._policy.neuron_retention

    @property
    def composer(self):
        return self._policy.composer

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy


@dataclass(frozen=True)
class cSET:
    """Periodic random rewiring with smallest-magnitude replacement."""

    event_interval: int = 500
    birth_budget: int = 2**31 - 1
    drop_fraction: float | Callable = 0.3
    freeze_event: int | None = None
    bounds_in: Bounds | None = None
    bounds_out: Bounds | None = None
    initial_weight: float = 0.0
    composer: None = field(default=None, init=False)
    profit: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        policy = Policy(
            schedule=PeriodicSchedule(
                event_interval=self.event_interval,
                birth_budget=self.birth_budget,
                freeze_event=self.freeze_event,
            ),
            proposers=(
                UniformEntryBirth(
                    self.bounds_in, self.bounds_out, self.initial_weight
                ),
            ),
            allocator=EvenBudgetAllocator(replacement_only=True),
            retention=MagnitudeCourt(self.drop_fraction),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)

    @property
    def schedule(self):
        return self._policy.schedule

    @property
    def proposers(self):
        return self._policy.proposers

    @property
    def allocator(self):
        return self._policy.allocator

    @property
    def retention(self):
        return self._policy.retention

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy


@dataclass(frozen=True)
class cRigL:
    """Gradient-greedy vertex-buying control arm, retained as a losing baseline."""

    event_interval: int = 500
    observe_window: int = 1
    birth_budget: int = 2**31 - 1
    drop_fraction: float | Callable = 0.3
    freeze_event: int | None = None
    pool_size: int = 4096
    pool: int | None = None
    decay: float = 0.9
    initial_weight: float = 0.0
    composer: None = field(default=None, init=False)
    profit: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.pool is not None and self.pool_size != 4096:
            raise ValueError("specify only one of pool and pool_size")
        effective_pool = self.pool_size if self.pool is None else self.pool
        object.__setattr__(self, "pool_size", effective_pool)
        policy = Policy(
            schedule=PeriodicSchedule(
                event_interval=self.event_interval,
                birth_budget=self.birth_budget,
                freeze_event=self.freeze_event,
                observe_window=self.observe_window,
            ),
            proposers=(
                GradFieldTopKBirth(
                    initial_weight=self.initial_weight,
                    decay=self.decay,
                    pool_size=effective_pool,
                ),
            ),
            allocator=EvenBudgetAllocator(replacement_only=True),
            retention=MagnitudeCourt(self.drop_fraction),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)

    @property
    def schedule(self):
        return self._policy.schedule

    @property
    def proposers(self):
        return self._policy.proposers

    @property
    def allocator(self):
        return self._policy.allocator

    @property
    def retention(self):
        return self._policy.retention

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy
