"""Lawful chart proposal — the initialization side of the operating envelope.

Where :mod:`torchcst.representation.survey` diagnoses an existing chart,
:func:`propose_chart` constructs one that is lawful *by construction*: the
box is sized so that neuron spacing lands in the measured window for its
dimension, and the returned :class:`Box` is directly the initialization --
drawing neuron coordinates and atom coordinates uniformly from it
(``box.sample``) reproduces the measured winning placement, which beat both
grids and similarity-derived embeddings on every seed tested.

**The proposal is written in spacing, not extent.**  dim-economy DIM-K2
(sibling ``cst`` repo, 105 arms at fixed atom count) measured neuron spacing
over σ to be what orders accuracy: inside the per-dimension window coverage
ranged over four orders of magnitude and per-axis extent over three with no
effect, and leaving the window collapsed every dimension.  There is no single
lawful extent -- at its own best spacing dim 1 wants 1434σ per axis and dim 6
wants 6.3σ, and both train -- so extent is derived here rather than chosen,
via ``extent = spacing * population**(1/dim)``.

Two defaults changed with that measurement and are worth stating plainly:

* ``dim`` no longer grows with population.  Distinguishability is a spacing
  property, so every dimension can hold any population; DIM-K2's accuracy
  peak sits at dim 3 (with 1, 2 and 4 within ~1-2 sd), and the default is
  :data:`DEFAULT_DIM`.  Atom cost is ``2*dim + 1`` per atom, so a
  parameter-tight caller should prefer a lower dimension -- the same sweep's
  parameter-matched series ordered dim 1 first.
* ``recommended_atoms`` is the population, not the cell count.  The old
  one-atom-per-cell target over-provisions by up to 64x at these dimensions;
  DIM-K2's best arms ran at one atom per neuron, down to coverage 0.016.

A proposed-and-sampled chart needs no survey; the round-trip
(propose → sample → survey finds nothing to flag) is pinned by a contract
test.  Every default is an argument: override ``spacing``, ``dim`` or
``coverage_floor`` to recalibrate for another kernel, domain shape, or
training timescale.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domains import Box
from .survey import COVERAGE_FLOOR, RECOMMENDED_SPACING, spacing_window

__all__ = ["ChartProposal", "propose_chart"]

#: **Superseded by** :data:`~torchcst.representation.survey.RECOMMENDED_SPACING`.
#: The old fixed per-axis extent target; kept because callers import it.
LAWFUL_AXIS_TARGET = 10.0

#: Default coordinate dimension.  DIM-K2's accuracy peak at fixed atom count,
#: with dims 1, 2 and 4 within roughly one to two standard deviations of it --
#: so this is a default, not a law.  Atom parameter cost is ``2*dim + 1``.
DEFAULT_DIM = 3


@dataclass(frozen=True)
class ChartProposal:
    """A lawful chart initialization, ready to execute.

    ``box`` is the proposed coordinate domain; sample neuron and atom
    coordinates uniformly from it.  ``recommended_atoms`` is one atom per
    neuron -- DIM-K2's best arms ran there -- and is a floor with growth
    headroom, not a ceiling.  ``spacing`` is the neuron spacing in σ units the
    box realises, and is the quantity the proposal is actually built around;
    ``cells`` is reported because callers still read it, but it is derived.
    """

    box: Box
    dim: int
    cells: float
    recommended_atoms: int
    notes: tuple[str, ...]
    spacing: float = float("nan")


def propose_chart(
    population: int,
    sigma: float,
    *,
    dim: int | None = None,
    atoms: int | None = None,
    spacing: float = RECOMMENDED_SPACING,
    coverage_floor: float = COVERAGE_FLOOR,
) -> ChartProposal:
    """Propose a lawful chart for ``population`` neurons at bandwidth ``sigma``.

    The box is sized so that neuron spacing lands on ``spacing`` σ:
    ``extent = spacing * sigma * population**(1/dim)`` per axis.  ``spacing``
    defaults to the one value that trained at every dimension DIM-K2 measured;
    ``dim`` defaults to :data:`DEFAULT_DIM`, that sweep's accuracy peak.

    Pass ``dim`` to override the dimension, ``spacing`` to sit elsewhere in
    the measured window, and ``atoms`` to have a planned atom count checked.
    Disagreements are reported in ``notes``, never raised.
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
    if not spacing > 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if not coverage_floor > 0.0:
        raise ValueError("coverage_floor must be positive")

    notes: list[str] = []
    chosen = DEFAULT_DIM if dim is None else dim
    axis_extent = spacing * population ** (1.0 / chosen)
    cells = axis_extent**chosen
    recommended = population

    window_low, window_high = spacing_window(chosen)
    if not window_low <= spacing <= window_high:
        side = "below" if spacing < window_low else "above"
        notes.append(
            f"spacing {spacing:.2f}σ is {side} the measured "
            f"{window_low:.1f}–{window_high:.1f}σ window for dim={chosen} — "
            "the chart is being built outside the band where charts trained"
        )
    if atoms is not None and atoms < population:
        notes.append(
            f"planned atoms {atoms} < population {population} — fewer atoms "
            f"than neurons; recommend ≥ {recommended}"
        )
    box = Box(0.0, axis_extent * float(sigma), chosen)
    return ChartProposal(
        box=box,
        dim=chosen,
        cells=cells,
        recommended_atoms=recommended,
        notes=tuple(notes),
        spacing=float(spacing),
    )
