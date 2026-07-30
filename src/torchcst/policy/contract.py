"""Loss-blind structural policy contracts for CST-native v4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from torchcst._validation import require_int
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
        require_int(self.update_step, "update_step", minimum=0)
        require_int(self.event_index, "event_index", minimum=0)


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
        require_int(self.event_index, "event_index", minimum=0)
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
            require_int(getattr(self, name), name, minimum=0)
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
        require_int(self.pool_size, "instrument pool_size", minimum=1)
        require_int(self.rank, "instrument rank", minimum=1)
        # Accepted spelling variant; the canonical form is the stored one.
        if self.aggregation == "abs-after-sum":
            object.__setattr__(self, "aggregation", "abs_after_sum")
        if self.aggregation != "abs_after_sum":
            raise ValueError("only abs_after_sum aggregation is supported")
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
        require_int(budget, "budget", minimum=0)
        requests = tuple(requests)
        allocations = [0] * len(requests)
        remaining = budget
        # Round-robin, one unit per pass, so leftover units land evenly.
        # Prune-kind requests (and every request under replacement_only) are
        # capped at their own replacement count; others may take the whole
        # budget. The progressed flag ends the loop once every request is at
        # its cap.
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
        require_int(budget, "budget", minimum=0)
        accepted: list[ProposalBundle] = []
        remaining = budget
        for bundle in tuple(bundles):
            if not isinstance(bundle, ProposalBundle):
                raise TypeError("bundles must contain ProposalBundle values")
            if not bundle.atomic:
                raise ValueError("only atomic proposal bundles are supported")
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
