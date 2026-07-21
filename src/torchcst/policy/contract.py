"""Loss-blind structural policy contracts for CST-native v4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from torchcst.storage import (
    NeuronRetire,
    NeuronView,
    SynapseBirth,
    SynapseDeath,
    SynapseView,
)

from .bundle import BundleComposer, ProposalBundle, bundle_birth_count
from .registry import RetiredCandidateRegistry


@dataclass(frozen=True)
class Clock:
    """Structural time measured in optimizer updates and structural events."""

    update_step: int
    event_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("update_step", self.update_step),
            ("event_index", self.event_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


class Phase(str, Enum):
    """Structural tempos owned by a schedule."""

    GROW = "grow"
    SWEEP = "sweep"
    FROZEN = "frozen"
    RESPONSE = "response"


@dataclass(frozen=True)
class EventDirective:
    """A schedule-issued event and its only source of birth supply."""

    event_index: int
    birth_budget: int
    phase: Phase
    ungate_budget: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.birth_budget, bool) or not isinstance(
            self.birth_budget, int
        ):
            raise TypeError("birth_budget must be an int")
        if self.birth_budget < 0:
            raise ValueError("birth_budget must be non-negative")
        if isinstance(self.ungate_budget, bool) or not isinstance(
            self.ungate_budget, int
        ):
            raise TypeError("ungate_budget must be an int")
        if self.ungate_budget < 0:
            raise ValueError("ungate_budget must be non-negative")


@dataclass(frozen=True)
class InstrumentSpec:
    """Declarative request for one engine-owned observation instrument."""

    name: str
    decay: float = 0.9
    pool_size: int = 4096
    aggregation: str = "abs_after_sum"
    rank: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("instrument name must be a non-empty string")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("instrument decay must be in [0, 1]")
        if isinstance(self.pool_size, bool) or not isinstance(self.pool_size, int):
            raise TypeError("instrument pool_size must be an int")
        if self.pool_size <= 0:
            raise ValueError("instrument pool_size must be positive")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("instrument rank must be an int")
        if self.rank <= 0:
            raise ValueError("instrument rank must be positive")
        if self.aggregation == "abs-after-sum":
            object.__setattr__(self, "aggregation", "abs_after_sum")
        if self.aggregation != "abs_after_sum":
            raise ValueError("step 4 supports only abs_after_sum aggregation")


@runtime_checkable
class Schedule(Protocol):
    """Clock-only structural event source."""

    def phase(self, clock: Clock) -> Phase: ...

    def event(self, clock: Clock) -> EventDirective | None: ...

    def observing(self, clock: Clock) -> bool: ...


@runtime_checkable
class OpProposer(Protocol):
    """Propose births from a view, an issued budget, and structural state."""

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]: ...


@dataclass(frozen=True)
class BudgetRequest:
    """One site/proposer recipient considered by a global allocator."""

    site: str
    proposer_index: int
    replacement_count: int


@runtime_checkable
class BudgetAllocator(Protocol):
    """Split one schedule-issued budget over proposal recipients."""

    def allocate(
        self, budget: int, requests: Sequence[BudgetRequest]
    ) -> tuple[int, ...]: ...

    def allocate_bundles(
        self, budget: int, bundles: Sequence[ProposalBundle]
    ) -> tuple[ProposalBundle, ...]: ...


@dataclass(frozen=True)
class EvenBudgetAllocator:
    """Deterministically share budget; optionally cap births by deaths."""

    replacement_only: bool = False

    def allocate(
        self, budget: int, requests: Sequence[BudgetRequest]
    ) -> tuple[int, ...]:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an int")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        requests = tuple(requests)
        allocations = [0] * len(requests)
        remaining = budget
        while remaining:
            progressed = False
            for index, request in enumerate(requests):
                cap = request.replacement_count if self.replacement_only else budget
                if allocations[index] >= cap:
                    continue
                allocations[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break
        return tuple(allocations)

    def allocate_bundles(
        self, budget: int, bundles: Sequence[ProposalBundle]
    ) -> tuple[ProposalBundle, ...]:
        """Accept whole bundles in stable order while the budget permits."""
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an int")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        accepted: list[ProposalBundle] = []
        remaining = budget
        for bundle in tuple(bundles):
            if not isinstance(bundle, ProposalBundle):
                raise TypeError("bundles must contain ProposalBundle values")
            if not bundle.atomic:
                raise ValueError("step 6 accepts only atomic proposal bundles")
            cost = bundle_birth_count(bundle)
            if cost <= remaining:
                accepted.append(bundle)
                remaining -= cost
        return tuple(accepted)


@runtime_checkable
class RetentionCourt(Protocol):
    """Return death operations using structural state only."""

    immunity_events: int

    def decide(
        self, view: SynapseView | NeuronView, ages: torch.Tensor, clock: Clock
    ) -> tuple[SynapseDeath | NeuronRetire, ...]: ...


@dataclass(frozen=True)
class Policy:
    """One swappable structural learning rule."""

    schedule: Schedule
    proposers: tuple[OpProposer, ...]
    allocator: BudgetAllocator
    retention: RetentionCourt
    composer: BundleComposer | None = None
    profit: Any | None = None
    neuron_retention: RetentionCourt | None = None

    @property
    def requires(self) -> tuple[InstrumentSpec, ...]:
        """Deduplicate every component's requests while preserving order."""
        components = (
            self.schedule,
            *self.proposers,
            self.allocator,
            self.retention,
            self.composer,
            self.profit,
            self.neuron_retention,
        )
        result: list[InstrumentSpec] = []
        for component in components:
            if component is None:
                continue
            requested = tuple(getattr(component, "requires", ()))
            if not all(isinstance(spec, InstrumentSpec) for spec in requested):
                raise TypeError("component requires must contain InstrumentSpec values")
            for spec in requested:
                if spec not in result:
                    result.append(spec)
        return tuple(result)
