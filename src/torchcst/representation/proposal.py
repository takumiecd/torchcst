"""Lawful chart proposal — the initialization side of the operating envelope.

Where :mod:`torchcst.representation.survey` diagnoses an existing chart,
:func:`propose_chart` constructs one that is lawful *by construction*: every
axis spans the measured extent peak, the dimension is the smallest that can
distinguish the population, and the returned :class:`Box` is directly the
initialization -- drawing neuron coordinates and atom coordinates uniformly
from it (``box.sample``) reproduces the measured winning placement, which
beat both grids and similarity-derived embeddings on every seed tested.

A proposed-and-sampled chart needs no survey; the round-trip
(propose → sample → survey finds nothing to flag) is pinned by a contract
test.  Defaults carry the measured envelope (FC-4 appendix J; the FC-7a3/a4
bracket; fc7h's coverage collapse -- reports in the sibling ``cst``
repository) and every one of them is an argument: override ``axis_extent``
or ``coverage_floor`` to recalibrate for another kernel, domain shape, or
training timescale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domains import Box
from .survey import COVERAGE_FLOOR

__all__ = ["ChartProposal", "propose_chart"]

#: Default per-axis extent (in σ units): the measured peak for
#: multi-dimensional hidden charts, bracketed on both sides (5σ too narrow,
#: 20σ collapsed).
LAWFUL_AXIS_TARGET = 10.0


@dataclass(frozen=True)
class ChartProposal:
    """A lawful chart initialization, ready to execute.

    ``box`` is the proposed coordinate domain; sample neuron and atom
    coordinates uniformly from it.  ``recommended_atoms`` targets one atom
    per resolvable cell; the measured capacity ladder was still unsaturated
    at ~1.3 atoms/cell, so treat it as a floor with growth headroom, not a
    ceiling.
    """

    box: Box
    dim: int
    cells: float
    recommended_atoms: int
    notes: tuple[str, ...]


def propose_chart(
    population: int,
    sigma: float,
    *,
    dim: int | None = None,
    atoms: int | None = None,
    axis_extent: float = LAWFUL_AXIS_TARGET,
    coverage_floor: float = COVERAGE_FLOOR,
) -> ChartProposal:
    """Propose a lawful chart for ``population`` neurons at bandwidth ``sigma``.

    Every axis spans ``axis_extent``σ (default: the measured 10σ/axis peak),
    and the dimension is the smallest ``d`` whose ``axis_extent^d``
    resolvable cells hold the population -- distinguishability requires
    cells ≥ population, and each added axis multiplies the atom budget
    needed for coverage, so the smallest sufficient dimension is the
    affordable one (the same cells-vs-coverage economy that ruled out both
    the 1-D line for 64 roles and the 3-D box for a 64--128 atom budget).

    Pass ``dim`` to override the dimension choice and ``atoms`` to have a
    planned atom count checked against ``coverage_floor``; disagreements are
    reported in ``notes``, never raised.
    """
    if isinstance(population, bool) or not isinstance(population, int):
        raise TypeError("population must be an int")
    if population <= 0:
        raise ValueError(f"population must be positive, got {population}")
    if not sigma > 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int)):
        raise TypeError("dim must be an int")
    if dim is not None and dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if atoms is not None and (isinstance(atoms, bool) or not isinstance(atoms, int)):
        raise TypeError("atoms must be an int")
    if atoms is not None and atoms <= 0:
        raise ValueError(f"atoms must be positive, got {atoms}")
    if not axis_extent > 1.0:
        raise ValueError(f"axis_extent must exceed 1 (got {axis_extent})")
    if not coverage_floor > 0.0:
        raise ValueError("coverage_floor must be positive")

    notes: list[str] = []
    sufficient = max(1, math.ceil(math.log(population, axis_extent)))
    chosen = dim if dim is not None else sufficient
    cells = axis_extent**chosen
    if population > cells:
        notes.append(
            f"population {population} exceeds ~{cells:.0f} resolvable cells at "
            f"dim={chosen} — roles cannot be distinguished; the smallest "
            f"sufficient dimension is {sufficient}"
        )
    recommended = max(population, math.ceil(cells))
    if atoms is not None and atoms < coverage_floor * cells:
        notes.append(
            f"planned atoms {atoms} give coverage {atoms / cells:.2f} < "
            f"{coverage_floor} — coverage-starved; recommend ≥ {recommended}"
        )
    box = Box(0.0, axis_extent * float(sigma), chosen)
    return ChartProposal(
        box=box,
        dim=chosen,
        cells=cells,
        recommended_atoms=recommended,
        notes=tuple(notes),
    )
