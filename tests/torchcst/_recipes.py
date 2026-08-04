"""Test-suite scaffolding: the retired 5c ``LC`` preset.

The mechanism tests use LC as a convenient, well-understood harness policy
(windowed uniform growth, rent-based cleanup, freeze).  It was removed from
the public recipe catalog in 2026-08 -- this private copy is a test fixture,
not an API, and its constants are pinned only by the tests that use it.
"""

from __future__ import annotations

from torchcst.policy.cadences import BirthWindowCadence
from torchcst.policy.contract import EvenBudgetDistributor, StructuralQuota
from torchcst.policy.courts import RentCourt
from torchcst.policy.families import SynapseLifecycle
from torchcst.policy.proposers import Bounds, UniformEntryBirth
from torchcst.policy.quotas import QuotaWindow, WindowedQuota
from torchcst.policy.tree import QuotaRegime


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
