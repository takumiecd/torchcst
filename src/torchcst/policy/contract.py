"""Loss-blind structural policy contracts for CST-native v4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from torchcst.storage import (
    NeuronKick,
    NeuronRetire,
    NeuronUngate,
    NeuronView,
    SynapseBirth,
    SynapseDeath,
    SynapseKick,
    SynapseMerge,
    SynapseView,
)

from .bundle import BundleComposer, Op, ProposalBundle, bundle_birth_count
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
    """Legacy schedule event carrying both timing and structural supply."""

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
    neuron_birth: int = 0
    synapse_prune: int | None = None
    neuron_prune: int | None = None

    def __post_init__(self) -> None:
        for name in ("synapse_birth", "synapse_merge", "neuron_birth"):
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
class Schedule(Protocol):
    """Legacy clock source that also carries operation budgets."""

    def phase(self, clock: Clock) -> Phase: ...

    def event(self, clock: Clock) -> EventDirective | None: ...

    def observing(self, clock: Clock) -> bool: ...


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


# Compatibility names for the pre-cadence public API. These are aliases, not
# storage allocators; physical slots remain exclusively owned by SlotPool.
BudgetAllocator = BudgetDistributor
EvenBudgetAllocator = EvenBudgetDistributor


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

    schedule: Schedule | None = None
    proposers: tuple[OpProposer, ...] = ()
    allocator: BudgetDistributor | None = None
    retention: RetentionCourt | None = None
    composer: BundleComposer | None = None
    profit: Any | None = None
    neuron_retention: RetentionCourt | None = None
    cadence: Cadence | None = None
    quota: QuotaPolicy | None = None
    distributor: BudgetDistributor | None = None
    observations: tuple[InstrumentRequirement, ...] = ()

    def __post_init__(self) -> None:
        if (self.schedule is None) == (self.cadence is None):
            raise ValueError("specify exactly one of schedule= or cadence=")
        if self.cadence is not None and self.quota is None:
            raise ValueError("cadence= requires a separate quota=")
        if self.schedule is not None and self.quota is not None:
            raise ValueError("legacy schedule= already owns its operation budget")
        if (self.allocator is None) == (self.distributor is None):
            raise ValueError("specify exactly one of allocator= or distributor=")
        if self.retention is None:
            raise ValueError("retention is required")
        if not isinstance(self.proposers, tuple):
            raise TypeError("proposers must be a tuple")
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")

    @property
    def active_cadence(self) -> Schedule | Cadence:
        """Return the configured timing source across old and new APIs."""
        cadence = self.cadence if self.cadence is not None else self.schedule
        assert cadence is not None
        return cadence

    @property
    def active_distributor(self) -> BudgetDistributor:
        """Return the configured logical distributor across both API names."""
        distributor = (
            self.distributor if self.distributor is not None else self.allocator
        )
        assert distributor is not None
        return distributor

    @property
    def requires(self) -> tuple[InstrumentRequirement, ...]:
        """Deduplicate every component's requests while preserving order."""
        components = (
            self.active_cadence,
            self.quota,
            *self.proposers,
            self.active_distributor,
            self.retention,
            self.composer,
            self.profit,
            self.neuron_retention,
        )
        result: list[InstrumentRequirement] = list(self.observations)
        if not all(
            isinstance(spec, (InstrumentSpec, ObservationRequest)) for spec in result
        ):
            raise TypeError(
                "observations must contain InstrumentSpec or ObservationRequest values"
            )
        for component in components:
            if component is None:
                continue
            requested = tuple(getattr(component, "requires", ()))
            if not all(
                isinstance(spec, (InstrumentSpec, ObservationRequest))
                for spec in requested
            ):
                raise TypeError(
                    "component requires must contain InstrumentSpec or "
                    "ObservationRequest values"
                )
            for spec in requested:
                if spec not in result:
                    result.append(spec)
        return tuple(result)


@dataclass(frozen=True)
class PolicyContext:
    """Read-only engine state presented to a whole-policy implementation."""

    clock: Clock
    synapses: Mapping[str, SynapseView]
    neurons: Mapping[str, NeuronView]
    ages: Mapping[str, torch.Tensor]
    instruments: Mapping[str, Mapping[str, Any]]
    registry: RetiredCandidateRegistry
    rng: torch.Generator

    def instrument(self, site: str, name: str) -> Any:
        """Return one declared site instrument with a useful lookup error."""
        try:
            return self.instruments[site][name]
        except KeyError as exc:
            raise KeyError(
                f"instrument {name!r} is not available at site {site!r}"
            ) from exc


@dataclass(frozen=True)
class StructuralPlan:
    """Independent operations or atomic bundles emitted by one policy event."""

    proposals: tuple[Op | ProposalBundle, ...] = ()
    synapse_immunity_events: int = 0
    neuron_immunity_events: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.proposals, tuple):
            raise TypeError("proposals must be a tuple")
        operation_types = (
            SynapseBirth,
            SynapseDeath,
            SynapseMerge,
            SynapseKick,
            NeuronUngate,
            NeuronRetire,
            NeuronKick,
        )
        if not all(
            isinstance(item, (*operation_types, ProposalBundle))
            for item in self.proposals
        ):
            raise TypeError("proposals must contain operations or ProposalBundle values")
        for name, value in (
            ("synapse_immunity_events", self.synapse_immunity_events),
            ("neuron_immunity_events", self.neuron_immunity_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@runtime_checkable
class StructuralPolicy(Protocol):
    """First-class extension point for implementing a policy as one object.

    Unlike :class:`Policy`, this contract does not require a schedule,
    allocator, proposer, or court decomposition. Returning ``None`` from
    :meth:`plan` means that no structural event occurs at that update; an empty
    plan is still an event and advances structural time.
    """

    requires: tuple[InstrumentRequirement, ...]

    def capture(self, clock: Clock) -> bool: ...

    def plan(self, context: PolicyContext) -> StructuralPlan | None: ...
