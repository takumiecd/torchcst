"""Step-2 policy catalog: lifecycle champion and cSET baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .bundle import BundleComposer
from .cadences import BirthWindowCadence, PeriodicCadence
from .contract import ActionSpec, EvenBudgetDistributor, Policy, StructuralQuota
from .courts import MagnitudeCourt, RentCourt
from .proposers import (
    Bounds,
    GradFieldTopKBirth,
    IncidentOutputBirth,
    MergeProposer,
    OrthogonalBirth,
    UniformBirth,
    UniformEntryBirth,
)
from .profit import ProfitCourt
from .quotas import ConstantQuota, QuotaWindow, WindowedQuota


class _PolicyAdapter:
    """Expose one built :class:`Policy` without repeating forwarding boilerplate."""

    _policy: Policy

    @property
    def schedule(self):
        """Compatibility alias for the catalog entry's budget-free cadence."""
        return self._policy.active_cadence

    @property
    def cadence(self):
        return self._policy.active_cadence

    @property
    def quota(self):
        return self._policy.quota

    @property
    def proposers(self):
        """Compatibility view of proposal-producing action rules."""
        return self._policy.proposal_rules

    @property
    def actions(self):
        return self._policy.active_actions

    @property
    def allocator(self):
        """Compatibility alias for the logical budget distributor."""
        return self._policy.active_distributor

    @property
    def distributor(self):
        return self._policy.active_distributor

    @property
    def retention(self):
        return self._policy.synapse_retention_rule

    @property
    def composer(self):
        return self._policy.composer

    @property
    def profit(self):
        return self._policy.profit

    @property
    def neuron_retention(self):
        return self._policy.neuron_retention_rule

    @property
    def requires(self):
        return self._policy.requires

    def as_policy(self) -> Policy:
        return self._policy


@dataclass(frozen=True)
class LC(_PolicyAdapter):
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
            cadence=BirthWindowCadence(
                event_interval=self.event_interval,
                birth_end_event=self.birth_end_event,
                freeze_event=self.freeze_event,
            ),
            quota=WindowedQuota(
                (
                    QuotaWindow(
                        self.birth_start_event,
                        self.birth_end_event,
                        StructuralQuota(synapse_birth=self.birth_budget),
                    ),
                ),
                default=StructuralQuota(),
            ),
            actions=(
                ActionSpec.synapse_prune(
                    RentCourt(
                        immunity_events=self.immunity_events,
                        rent_ratio=self.rent_ratio,
                        strikes=self.strikes,
                    )
                ),
                ActionSpec.synapse_birth(
                    UniformEntryBirth(
                        self.bounds_in, self.bounds_out, self.initial_weight
                    )
                ),
            ),
            distributor=EvenBudgetDistributor(),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class LC_anti(_PolicyAdapter):
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
            cadence=BirthWindowCadence(
                event_interval=self.event_interval,
                birth_end_event=self.birth_end_event,
                freeze_event=self.freeze_event,
                observe_window=self.observe_window,
            ),
            quota=WindowedQuota(
                (
                    QuotaWindow(
                        self.birth_start_event,
                        self.birth_end_event,
                        StructuralQuota(synapse_birth=self.birth_budget),
                    ),
                ),
                default=StructuralQuota(),
            ),
            actions=(
                ActionSpec.synapse_prune(
                    RentCourt(
                        immunity_events=self.immunity_events,
                        rent_ratio=self.rent_ratio,
                        strikes=self.strikes,
                    )
                ),
                ActionSpec.synapse_birth(
                    OrthogonalBirth(rank=self.certificate_rank)
                ),
            ),
            distributor=EvenBudgetDistributor(),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class LC_response(_PolicyAdapter):
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
            cadence=BirthWindowCadence(
                event_interval=self.event_interval,
                birth_end_event=self.birth_end_event,
                freeze_event=self.freeze_event,
                response_events=self.response_events,
            ),
            quota=WindowedQuota(
                (
                    QuotaWindow(
                        self.birth_start_event,
                        self.birth_end_event,
                        StructuralQuota(synapse_birth=self.birth_budget),
                    ),
                    QuotaWindow(
                        self.response_events[0],
                        self.response_events[1],
                        StructuralQuota(
                            synapse_birth=self.response_birth_budget,
                            neuron_birth=self.response_ungates_per_event,
                        ),
                    ),
                ),
                default=StructuralQuota(),
            ),
            actions=(
                ActionSpec.synapse_prune(retention),
                ActionSpec.neuron_prune(neuron_retention),
                ActionSpec.synapse_birth(
                    UniformEntryBirth(
                        self.bounds_in, self.bounds_out, self.initial_weight
                    )
                ),
                ActionSpec.synapse_birth(IncidentOutputBirth(self.initial_weight)),
            ),
            distributor=EvenBudgetDistributor(),
            composer=composer,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class LC_merge(_PolicyAdapter):
    """Rank-one lifecycle whose scheduled merge trials require ProfitCourt."""

    event_interval: int = 200
    birth_start_event: int = 1
    birth_end_event: int = 5
    birth_budget: int = 1
    freeze_event: int | None = 10
    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    similarity_threshold: float = 0.9
    min_profit: float = 0.0
    cost_rate: float = 0.0
    composer: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        policy = Policy(
            cadence=BirthWindowCadence(
                event_interval=self.event_interval,
                birth_end_event=self.birth_end_event,
                freeze_event=self.freeze_event,
            ),
            quota=WindowedQuota(
                (
                    QuotaWindow(
                        self.birth_start_event,
                        self.birth_end_event,
                        StructuralQuota(synapse_merge=self.birth_budget),
                    ),
                ),
                default=StructuralQuota(),
            ),
            actions=(
                ActionSpec.synapse_prune(
                    RentCourt(
                        immunity_events=self.immunity_events,
                        rent_ratio=self.rent_ratio,
                        strikes=self.strikes,
                    )
                ),
                ActionSpec.synapse_merge(
                    MergeProposer(self.similarity_threshold)
                ),
            ),
            distributor=EvenBudgetDistributor(),
            composer=None,
            profit=ProfitCourt(
                min_profit=self.min_profit,
                cost_rate=self.cost_rate,
            ),
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class GrowthByProfit(_PolicyAdapter):
    """Greedy profit-gated forward construction testing theory U-2.

    Every event proposes ``atoms_per_event`` candidate atoms and accepts them
    only if the realized loss reduction exceeds ``price`` (a ProfitCourt
    price in the *same units* as ``price_for``'s ``cost_rate * delta_params``
    — i.e. loss-units per resource-count unit, generically "one more
    parameter"). No rent/prune: this policy tests pure add-while-profitable
    construction (S-5 decisions only), not the 5c retention lifecycle, so a
    run's natural stopping point is theory U-2's predicted K*(price).

    This policy declares only a birth action because it has nothing to prune.
    """

    event_interval: int = 1
    atoms_per_event: int = 1
    price: float = 0.0
    min_profit: float = 0.0
    initial_weight: float = 0.0
    bounds_in: Bounds | None = None
    bounds_out: Bounds | None = None
    composer: None = field(default=None, init=False)
    _policy: Policy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        policy = Policy(
            cadence=PeriodicCadence(
                event_interval=self.event_interval,
                freeze_event=None,
            ),
            quota=ConstantQuota(
                StructuralQuota(synapse_birth=self.atoms_per_event)
            ),
            actions=(
                ActionSpec.synapse_birth(
                    UniformBirth(
                        bounds_in=self.bounds_in,
                        bounds_out=self.bounds_out,
                        initial_weight=self.initial_weight,
                    )
                ),
            ),
            distributor=EvenBudgetDistributor(),
            composer=None,
            profit=ProfitCourt(min_profit=self.min_profit, cost_rate=self.price),
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class cSET(_PolicyAdapter):
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
            cadence=PeriodicCadence(
                event_interval=self.event_interval,
                freeze_event=self.freeze_event,
            ),
            quota=ConstantQuota(
                StructuralQuota(synapse_birth=self.birth_budget)
            ),
            actions=(
                ActionSpec.synapse_prune(MagnitudeCourt(self.drop_fraction)),
                ActionSpec.synapse_birth(
                    UniformEntryBirth(
                        self.bounds_in, self.bounds_out, self.initial_weight
                    )
                ),
            ),
            distributor=EvenBudgetDistributor(replacement_only=True),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)


@dataclass(frozen=True)
class cRigL(_PolicyAdapter):
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
            cadence=PeriodicCadence(
                event_interval=self.event_interval,
                freeze_event=self.freeze_event,
                observe_window=self.observe_window,
            ),
            quota=ConstantQuota(
                StructuralQuota(synapse_birth=self.birth_budget)
            ),
            actions=(
                ActionSpec.synapse_prune(MagnitudeCourt(self.drop_fraction)),
                ActionSpec.synapse_birth(
                    GradFieldTopKBirth(
                        initial_weight=self.initial_weight,
                        decay=self.decay,
                        pool_size=effective_pool,
                    )
                ),
            ),
            distributor=EvenBudgetDistributor(replacement_only=True),
            composer=None,
            profit=None,
        )
        object.__setattr__(self, "_policy", policy)
