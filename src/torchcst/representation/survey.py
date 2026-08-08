"""Chart operating-envelope survey — an optional diagnostic.

A continuous chart only makes its parts work -- coordinate learning, scored
birth, growth -- when it is operated inside an empirical envelope.  The
2026-08 conv arc measured that envelope's boundaries the hard way, and this
module turns those lessons into an executable diagnostic: given a chart's
neuron coordinates and the kernel bandwidth, :func:`survey_chart` reports
the per-axis and whole-chart quantities the lessons are written in, and
flags the known failure regimes.

The survey is *optional*.  Charts built through
:func:`torchcst.representation.propose_chart` are lawful by construction and
need no survey; the instrument earns its keep on charts you did *not*
propose -- hand-built ones, data-pinned ones (pixels, conv taps), or charts
inherited from an experiment you are diagnosing.

The five regimes it detects, with their measured provenance (all in the
sibling ``cst`` repository's reports):

* **Quasi-discrete axis** -- neuron spacing ≳ 5σ.  The kernel cannot bridge
  neighbouring neurons, so a coordinate moving between them travels through
  data-free vacuum: position gradients carry no unique signal and atoms
  diffuse, flee, or freeze rather than localize (TGM-C0 ``ρ_partial = 0``;
  the birth-radius probe's mix/eject/freeze regimes).  Such an axis should be
  treated as discrete -- pinned, or handled by policy-side selection -- not
  learned by SGD.
* **Indistinguishable population** -- more neurons than the chart can
  resolve (population exceeds ``∏ extent_d/σ`` cells).  Neurons closer than
  ~σ share their kernel column and cannot be told apart; a 1-D channel axis
  holding 64 roles in a 10σ line is this failure (FC-7a's σ_c = 1.0 arms).
* **Out-of-window spacing** -- the primary gate, and the one the others
  turn out to be symptoms of.  dim-economy DIM-K2 (sibling ``cst`` repo:
  CIFAR-gray depth-3, atom count and everything else held fixed, 5
  dimensions x 7 spacings x 3 seeds) measured **neuron spacing over σ** to be
  what orders accuracy.  Inside the per-dimension band of
  :data:`SPACING_WINDOW`, coverage ranged over four orders of magnitude and
  per-axis extent over three with no effect; leaving the band collapsed every
  dimension.  The band narrows and its centre rises with dimension, because a
  σ-ball in d dimensions holds ~(σ/spacing)^d neurons.
  :data:`RECOMMENDED_SPACING` is the one value that trained at every measured
  dimension.
* **Coverage starvation** -- fewer atoms than resolvable cells.  **Demoted by
  DIM-K2**: charts trained down to coverage 0.016, thirty-two times under
  :data:`COVERAGE_FLOOR`, once their spacing was in window.  The collapses
  this regime was fitted to (fc7h's lawful-domain joint conv chart at K=256;
  the 20σ/axis box) are high-dimensional joint charts whose spacing had left
  a narrow band -- the atom budget was the symptom, the spacing the cause.
  The note now fires only alongside an out-of-window spacing, where chart
  factorization (volume as a sum, not a product) remains the fix.
* **Out-of-law extent** -- likewise demoted.  The capacity law
  ``capacity = extent/σ`` (FC-4) read a sweet spot near 10σ per axis, and
  data-pinned input charts ran best near 20--30σ, because both were measured
  at a population that put those extents at a lawful spacing.  DIM-K2 found
  no single lawful extent: at its own best spacing dim 1 wants 1434σ per axis
  and dim 6 wants 6.3σ, and both train.  An extent only means something once
  divided by ``population**(1/dim)`` -- that is, as a spacing.

Every threshold is a keyword argument whose default is the measured value;
override them freely -- experiments that intentionally probe outside the
envelope, or that recalibrate it for another kernel/domain, must remain
expressible.  The survey is deliberately *descriptive*: it returns numbers
and notes, and raises only on malformed input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "AxisSurvey",
    "ChartSurvey",
    "RECOMMENDED_SPACING",
    "SPACING_WINDOW",
    "spacing_window",
    "survey_chart",
]

#: Default neuron spacing (in σ units) beyond which an axis is effectively
#: discrete: the kernel factor across one gap is exp(-spacing²/2) ≤ 4e-6.
#: DIM-K2 corroborates the value from the other side: 4σ still trained at
#: dim ≤ 2 and 5.6σ trained at no dimension at all, so the hard pathology
#: boundary sits between them.
QUASI_DISCRETE_SPACING = 5.0

#: Measured neuron-spacing window, in σ units, keyed by coordinate dimension
#: (dim-economy DIM-K2, sibling ``cst`` repo: CIFAR-gray depth-3, H = K = 1024
#: held fixed, 5 dimensions × 7 spacings × 3 seeds = 105 arms; each entry is
#: the band of grid points that cleared the working-accuracy threshold).
#:
#: Spacing — not cells, coverage, chart extent, or dimension — is what orders
#: accuracy.  Inside these bands coverage ranged over four orders of magnitude
#: (0.016 → 2.0) and per-axis extent over three (6σ → 4096σ) with no effect,
#: while leaving the band collapsed every dimension.  The band **narrows** and
#: its centre **rises** with dimension: in d dimensions a σ-ball holds
#: ~(σ/spacing)^d neurons, so a spacing that keeps kernel columns distinct at
#: dim 1 makes them collinear at dim 6.
SPACING_WINDOW: dict[int, tuple[float, float]] = {
    1: (0.7, 4.0),
    2: (0.7, 4.0),
    3: (1.4, 2.8),
    4: (1.4, 2.8),
    6: (2.0, 2.8),
}

#: The one spacing that cleared the threshold at EVERY measured dimension --
#: the only grid point in DIM-K2 where dims 1, 2, 3, 4 and 6 all trained.  Use
#: it whenever the dimension is not known in advance.
RECOMMENDED_SPACING = 2.0

#: Default atoms-per-resolvable-cell ratio below which the chart is flagged
#: as coverage-starved.  **Demoted by DIM-K2**: coverage is a derived quantity,
#: not a gate.  Working charts were measured down to coverage 0.016 -- 32×
#: below this floor -- once their spacing was in window, so the note now fires
#: only for a chart that is coverage-poor *and* out of its spacing window.  The
#: fc7h / wide-box collapses this floor was fitted to are all high-dimensional
#: joint charts whose spacing had left the band; the floor was reading the
#: symptom.
COVERAGE_FLOOR = 0.5


#: Default advisory extent window (in σ units per axis) for multi-dimensional
#: hidden charts: the measured peak sits near 10σ/axis, bracketed by a
#: too-narrow 5σ and a coverage-collapsing 20σ (FC-4 appendix J; FC-7a3/a4).
#: **Superseded by :data:`SPACING_WINDOW`**, kept because callers pass it.
#: DIM-K2 measured no single lawful extent band: at the best spacing for its
#: dimension, dim 1 wants 1434σ per axis and dim 6 wants 6.3σ, both of which
#: train.  An extent is only meaningful once divided by ``population**(1/dim)``
#: -- that is, as a spacing.  The advisory extent note now fires only when the
#: axis is ALSO out of its spacing window.
LAWFUL_AXIS_EXTENT = (5.0, 20.0)


def spacing_window(dim: int) -> tuple[float, float]:
    """The measured ``(lo, hi)`` spacing band, in σ units, for ``dim``.

    Dimensions between measured points take the intersection of their two
    neighbours, and dimensions above the largest measured one inherit its
    band: both are the conservative reading, since the band only ever narrows
    as dimension grows.  The band is advisory in the same sense as the rest of
    this module -- it reports, it never raises.
    """
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an int")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if dim in SPACING_WINDOW:
        return SPACING_WINDOW[dim]
    measured = sorted(SPACING_WINDOW)
    if dim > measured[-1]:
        return SPACING_WINDOW[measured[-1]]
    below = max(d for d in measured if d < dim)
    above = min(d for d in measured if d > dim)
    lo_b, hi_b = SPACING_WINDOW[below]
    lo_a, hi_a = SPACING_WINDOW[above]
    return (max(lo_b, lo_a), min(hi_b, hi_a))


@dataclass(frozen=True)
class AxisSurvey:
    """One chart axis in σ units."""

    extent_over_sigma: float
    spacing_over_sigma: float
    unique_positions: int
    quasi_discrete: bool


@dataclass(frozen=True)
class ChartSurvey:
    """Whole-chart survey result.

    ``cells`` is the resolvable-cell estimate ``∏ max(1, extent_d/σ)`` --
    the capacity law's volume form.  ``coverage`` is ``atoms / cells`` when
    an atom count was supplied, else ``None``.  ``notes`` names every
    detected failure regime in plain language; an empty tuple means the
    survey found nothing to flag, not that the chart is guaranteed lawful.

    ``lattice_spacing_over_sigma`` is the quantity :data:`SPACING_WINDOW` is
    written in, and is deliberately **not** the same as an axis's
    ``spacing_over_sigma``.  An axis reports the median gap between its
    *marginal* coordinate values; for a lattice chart the two agree, but for a
    randomly-placed cloud every projected value is distinct, so the marginal
    gap shrinks like ``extent/N`` while the distance an actual neighbour sits
    at shrinks like ``extent/N**(1/dim)``.  The window is about neighbours, so
    it uses the lattice-equivalent form.  Both are reported: the marginal gap
    is the right reading for a data-pinned axis (conv taps), the lattice
    spacing for the chart as a whole.
    """

    sigma: float
    population: int
    axes: tuple[AxisSurvey, ...]
    cells: float
    coverage: float | None
    notes: tuple[str, ...]
    lattice_spacing_over_sigma: float = float("nan")
    in_spacing_window: bool = True

    @property
    def dim(self) -> int:
        return len(self.axes)


def _axis_survey(column: Tensor, sigma: float, discrete_spacing: float) -> AxisSurvey:
    unique = torch.unique(column)
    extent = float(unique.max() - unique.min()) if unique.numel() > 1 else 0.0
    if unique.numel() > 1:
        gaps = unique.sort().values.diff()
        spacing = float(gaps.median()) / sigma
    else:
        # A single occupied position has no gap; the axis is degenerate and
        # reported as maximally discrete rather than pretending a spacing.
        spacing = math.inf
    return AxisSurvey(
        extent_over_sigma=extent / sigma,
        spacing_over_sigma=spacing,
        unique_positions=int(unique.numel()),
        quasi_discrete=spacing >= discrete_spacing,
    )


def survey_chart(
    coordinates: Tensor,
    sigma: float,
    *,
    atoms: int | None = None,
    quasi_discrete_spacing: float = QUASI_DISCRETE_SPACING,
    lawful_extent: tuple[float, float] = LAWFUL_AXIS_EXTENT,
    coverage_floor: float = COVERAGE_FLOOR,
) -> ChartSurvey:
    """Survey one chart against the measured operating envelope.

    ``coordinates`` is the ``(N, D)`` neuron-coordinate tensor of the chart
    (the same tensor a chart hands to its kernel).  ``sigma`` is the kernel
    bandwidth that will act on it.  ``atoms`` is the planned live-atom count
    for the store reading this chart; when given, coverage is reported.

    The three thresholds default to the measured envelope (see the module
    docstring for provenance) and may be overridden per call.
    """
    if not isinstance(coordinates, Tensor):
        raise TypeError("coordinates must be a Tensor")
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must have shape (N, D) with N >= 1")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if not sigma > 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if atoms is not None and (isinstance(atoms, bool) or not isinstance(atoms, int)):
        raise TypeError("atoms must be an int")
    if atoms is not None and atoms <= 0:
        raise ValueError(f"atoms must be positive, got {atoms}")
    if not quasi_discrete_spacing > 0.0:
        raise ValueError("quasi_discrete_spacing must be positive")
    low, high = float(lawful_extent[0]), float(lawful_extent[1])
    if not 0.0 < low < high:
        raise ValueError("lawful_extent must be an increasing positive pair")
    if not coverage_floor > 0.0:
        raise ValueError("coverage_floor must be positive")

    coords = coordinates.detach().to(torch.float64)
    population = coords.shape[0]
    axes = tuple(
        _axis_survey(coords[:, d], float(sigma), quasi_discrete_spacing)
        for d in range(coords.shape[1])
    )

    cells = 1.0
    for axis in axes:
        cells *= max(1.0, axis.extent_over_sigma)
    coverage = None if atoms is None else atoms / cells

    # The lattice-equivalent neighbour distance: geometric-mean extent (in σ,
    # sharing `cells`' floor so the two never disagree) over population**(1/dim).
    # See ChartSurvey's docstring for why this is not an axis's marginal gap.
    dim = len(axes)
    lattice_spacing = cells ** (1.0 / dim) / population ** (1.0 / dim)
    window_low, window_high = spacing_window(dim)
    in_window = window_low <= lattice_spacing <= window_high

    notes: list[str] = []
    for d, axis in enumerate(axes):
        if axis.quasi_discrete and axis.unique_positions > 1:
            notes.append(
                f"axis {d}: spacing {axis.spacing_over_sigma:.1f}σ ≥ "
                f"{quasi_discrete_spacing:.0f}σ — quasi-discrete; coordinate "
                "learning on this axis moves through data-free vacuum "
                "(treat it discretely or pin it)"
            )
        elif axis.unique_positions == 1:
            notes.append(f"axis {d}: single occupied position — degenerate axis")
    if population > cells:
        notes.append(
            f"population {population} exceeds ~{cells:.0f} resolvable cells — "
            "neurons closer than σ share kernel columns and cannot be told "
            "apart (over-dense chart)"
        )
    if not in_window:
        side = "below" if lattice_spacing < window_low else "above"
        why = (
            "neurons sit inside one another's kernel support and share columns"
            if side == "below" else
            "kernel columns no longer overlap, so position gradients carry no "
            "unique signal"
        )
        notes.append(
            f"neuron spacing {lattice_spacing:.2f}σ is {side} the measured "
            f"{window_low:.1f}–{window_high:.1f}σ window for dim={dim} — {why}. "
            "Spacing is the gate; cells, coverage and extent are derived from it"
        )
    # Coverage and per-axis extent were the pre-DIM-K2 gates.  Both are now
    # advisory and fire only for a chart that is already out of its spacing
    # window, so they annotate a diagnosed failure instead of raising their own
    # false positives on the wide-but-correctly-spaced charts DIM-K2 measured
    # to train perfectly well.
    if not in_window and coverage is not None and coverage < coverage_floor:
        notes.append(
            f"coverage {coverage:.2f} atoms/cell < {coverage_floor} — "
            "coverage-starved as well; chart factorization (volume as a sum, "
            "not a product) is what makes the capacity law affordable"
        )
    if not in_window and len(axes) > 1:
        for d, axis in enumerate(axes):
            if axis.unique_positions <= 1 or axis.quasi_discrete:
                continue
            if axis.extent_over_sigma < low:
                notes.append(
                    f"axis {d}: extent {axis.extent_over_sigma:.1f}σ below the "
                    f"~{low:.0f}–{high:.0f}σ advisory window for hidden axes"
                )
            elif axis.extent_over_sigma > high:
                notes.append(
                    f"axis {d}: extent {axis.extent_over_sigma:.1f}σ above the "
                    f"~{low:.0f}–{high:.0f}σ advisory window — coverage and "
                    "birth economy degrade in wide boxes"
                )

    return ChartSurvey(
        sigma=float(sigma),
        population=population,
        axes=axes,
        cells=cells,
        coverage=coverage,
        notes=tuple(notes),
        lattice_spacing_over_sigma=lattice_spacing,
        in_spacing_window=in_window,
    )
