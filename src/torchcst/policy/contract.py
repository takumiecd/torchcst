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

from .bundle import ProposalBundle, bundle_birth_count
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
class EventSignal:
    """Cadence-issued structural time point with no resource budget."""

    event_index: int
    phase: Phase

    def __post_init__(self) -> None:
        if isinstance(self.event_index, bool) or not isinstance(self.event_index, int):
            raise TypeError("event_index must be an int")
        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")
        if not isinstance(self.phase, Phase):
            raise TypeError("phase must be a Phase")


@dataclass(frozen=True)
class StructuralQuota:
    """Logical operation limits, independent of physical storage placement."""

    synapse_birth: int = 0
    synapse_merge: int = 0
    synapse_absorb: int = 0
    neuron_birth: int = 0
    synapse_prune: int | None = None
    neuron_prune: int | None = None

    def __post_init__(self) -> None:
        for name in ("synapse_birth", "synapse_merge", "synapse_absorb", "neuron_birth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("synapse_prune", "neuron_prune"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int or None")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def zero(cls) -> StructuralQuota:
        """Return a quota that permits no structural operations."""
        return cls(synapse_prune=0, neuron_prune=0)

    def limit(self, kind: str) -> int | None:
        """Return one named logical limit for generic action routing."""
        try:
            return getattr(self, kind)
        except AttributeError as exc:
            raise KeyError(f"unknown structural quota kind {kind!r}") from exc


@dataclass(frozen=True)
class InstrumentSpec:
    """Declarative request for one engine-owned observation instrument."""

    name: str
    decay: float = 0.9
    pool_size: int = 4096
    aggregation: str = "abs_after_sum"
    rank: int = 1
    timing: str = "after_backward"

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
        if self.timing not in {"backward_inline", "after_backward"}:
            raise ValueError("instrument timing must be backward_inline or after_backward")


@runtime_checkable
class ObservationRequest(Protocol):
    """Third-party factory request for one engine-owned capture instrument."""

    name: str
    timing: Any

    def build(self, context: Any) -> Any: ...


InstrumentRequirement = InstrumentSpec | ObservationRequest


@runtime_checkable
class Cadence(Protocol):
    """Clock-only observation and event timing with no resource decisions."""

    def phase(self, clock: Clock) -> Phase: ...

    def event(self, clock: Clock) -> EventSignal | None: ...

    def observing(self, clock: Clock) -> bool: ...


@runtime_checkable
class QuotaPolicy(Protocol):
    """Produce logical structural limits for one cadence-issued event."""

    def at(self, clock: Clock, phase: Phase) -> StructuralQuota: ...


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
    """One site/action recipient considered by a logical distributor."""

    site: str
    proposer_index: int
    replacement_count: int
    kind: str = "synapse_birth"


@runtime_checkable
class BudgetDistributor(Protocol):
    """Split one logical quota over site/action recipients."""

    def allocate(
        self, budget: int, requests: Sequence[BudgetRequest]
    ) -> tuple[int, ...]: ...

    def allocate_bundles(
        self, budget: int, bundles: Sequence[ProposalBundle]
    ) -> tuple[ProposalBundle, ...]: ...


@dataclass(frozen=True)
class EvenBudgetDistributor:
    """Deterministically distribute quota; optionally cap births by deaths."""

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
                cap = (
                    request.replacement_count
                    if self.replacement_only or request.kind.endswith("_prune")
                    else budget
                )
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
