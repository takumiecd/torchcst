"""Named, validated tree assemblies (``docs/policy-tree-design.md`` Recipes).

A recipe is a plain function returning a ready-to-bind policy-tree root, one
per shipped configuration, and the catalog stays *small on purpose*: only
assemblies in active research use live here.  ``FastConstruction`` is the
FASTCON default (cSFW-grow into cVP-style operation).  The historical 5c
presets (``LC``, ``LC_response``, ``GrowthByProfit``) were removed from the
public surface in 2026-08; the mechanism test suite keeps its own private
copy of ``LC`` as scaffolding (``tests/torchcst/_recipes.py``), which is a
test fixture, not an API.
"""

from __future__ import annotations

from .cadences import BirthWindowCadence
from .contract import EvenBudgetDistributor, StructuralQuota
from .families import cSFW
from .profit import ProfitCourt
from .quotas import QuotaWindow, WindowedQuota
from .tree import QuotaRegime


def FastConstruction(
    *,
    event_interval: int = 200,
    growth_events: int = 5,
    atoms_per_event: int = 1,
    pool_size: int = 4096,
    multistart: int = 4,
    trust: float = 0.01,
    ridge: float = 1.0e-4,
    price: float = 0.0,
    min_profit: float = 0.0,
) -> QuotaRegime:
    """Validated fast-construction assembly: cSFW-grow, then cVP-only.

    The first ``growth_events`` root events admit pure tangent births.  The
    family-private refit rule starts on the following event and then solves
    all amplitudes at every root event; positions remain ordinary SGD
    parameters.  Every birth/refit is guarded by the root's actual-loss
    profit trial, as required for solved insertions.

    Deep callers must seed every layer with a small live nucleus before
    binding this recipe: an empty upstream layer has zero construction
    gradient and cannot bootstrap itself.
    """
    method = cSFW(
        polish_iters=0,
        backfit="event",
        backfit_start_event=growth_events,
        backfit_position_iters=0,
        backfit_consume=True,
        pool_size=pool_size,
        multistart=multistart,
        trust=trust,
        ridge=ridge,
    )
    return QuotaRegime(
        budget=atoms_per_event,
        method=method,
        cadence=BirthWindowCadence(
            event_interval=event_interval,
            birth_end_event=growth_events,
            freeze_event=None,
            observe_window=event_interval,
        ),
        quota=WindowedQuota(
            (
                QuotaWindow(
                    1,
                    growth_events,
                    StructuralQuota(synapse_birth=atoms_per_event),
                ),
            ),
            default=StructuralQuota(),
        ),
        distributor=EvenBudgetDistributor(),
        profit=ProfitCourt(min_profit=min_profit, cost_rate=price),
    )
