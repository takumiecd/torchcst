"""Named, validated tree assemblies (``docs/policy-tree-design.md`` Recipes).

A recipe is a plain function returning a ready-to-bind policy-tree root: the
frozen catalog presets re-expressed in the tree vocabulary, one per shipped
configuration. Recipes carry the discipline the design note registered --
**recipe = 検証済み構成** -- and stay few (既定1＋対照2 per family).

Phase 2 S4 retires the composed-``Policy`` presets in ``catalog.py``; each
one that survives does so as a recipe here, gated by the S0 statistical
fingerprint (``tools/phase2_baseline.json``): the recipe must reproduce the
preset's behavior within the preregistered tolerances before the preset is
deleted.
"""

from __future__ import annotations

from .cadences import BirthWindowCadence, PeriodicCadence
from .contract import EvenBudgetDistributor, StructuralQuota
from .courts import RentCourt
from .families import SynapseLifecycle
from .profit import ProfitCourt
from .proposers import Bounds, UniformBirth, UniformEntryBirth
from .quotas import QuotaWindow, WindowedQuota
from .tree import QuotaRegime


def LC(
    *,
    event_interval: int = 200,
    birth_start_event: int = 1,
    birth_end_event: int = 5,
    birth_budget: int = 5,
    freeze_event: int | None = 10,
    immunity_events: int = 3,
    rent_ratio: float = 0.3,
    strikes: int = 2,
    bounds_in: Bounds | None = None,
    bounds_out: Bounds | None = None,
    initial_weight: float = 0.0,
) -> QuotaRegime:
    """The 5c lifecycle champion as a tree: windowed uniform growth, then a
    rent-based cleanup sweep, then freeze.

    Same constants, same parts as the retired ``catalog.LC`` preset --
    ``UniformEntryBirth`` growth inside the birth window, ``RentCourt``
    retention throughout -- assembled as a quota family whose root owns the
    cadence (``BirthWindowCadence``) and the time-varying supply
    (``WindowedQuota``: the birth budget exists only inside the window).
    """

    def make_birth(lam: float | None) -> UniformEntryBirth:
        del lam
        return UniformEntryBirth(bounds_in, bounds_out, initial_weight)

    method = SynapseLifecycle(
        birth_factory=make_birth,
        prune_factory=lambda: RentCourt(
            immunity_events=immunity_events,
            rent_ratio=rent_ratio,
            strikes=strikes,
        ),
        priceable=False,
        label="LC",
    )
    return QuotaRegime(
        budget=birth_budget,
        method=method,
        cadence=BirthWindowCadence(
            event_interval=event_interval,
            birth_end_event=birth_end_event,
            freeze_event=freeze_event,
        ),
        quota=WindowedQuota(
            (
                QuotaWindow(
                    birth_start_event,
                    birth_end_event,
                    StructuralQuota(synapse_birth=birth_budget),
                ),
            ),
            default=StructuralQuota(),
        ),
        distributor=EvenBudgetDistributor(),
    )


def GrowthByProfit(
    *,
    event_interval: int = 1,
    atoms_per_event: int = 1,
    price: float = 0.0,
    min_profit: float = 0.0,
    initial_weight: float = 0.0,
    bounds_in: Bounds | None = None,
    bounds_out: Bounds | None = None,
) -> QuotaRegime:
    """Greedy profit-gated forward construction (theory U-2) as a tree.

    Same parts as the retired ``catalog.GrowthByProfit`` preset: every event
    proposes ``atoms_per_event`` uniform candidates and the root's profit
    court keeps them only when the realized loss reduction beats the price.
    No prune -- the run's natural stop is U-2's predicted K*(price). The
    trial itself is the root's own subprotocol (docs/policy-tree-phase2.md
    ruling 2); the engine only lends its checkpoint mechanism.
    """

    def make_birth(lam: float | None) -> UniformBirth:
        del lam
        return UniformBirth(
            bounds_in=bounds_in,
            bounds_out=bounds_out,
            initial_weight=initial_weight,
        )

    method = SynapseLifecycle(
        birth_factory=make_birth,
        priceable=False,
        label="GrowthByProfit",
    )
    return QuotaRegime(
        budget=atoms_per_event,
        method=method,
        cadence=PeriodicCadence(event_interval=event_interval, freeze_event=None),
        distributor=EvenBudgetDistributor(),
        profit=ProfitCourt(min_profit=min_profit, cost_rate=price),
    )
