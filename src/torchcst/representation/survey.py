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

The four regimes it detects, with their measured provenance (all in the
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
* **Coverage starvation** -- fewer atoms than resolvable cells.  Widening a
  chart to satisfy the capacity law multiplies cells *per axis*; a joint
  chart pays the product and its atom budget stops covering the volume
  (fc7h: the lawful-domain joint conv chart collapsed at K=256; the 20σ/axis
  box collapse).  The capacity law is only affordable together with chart
  factorization, which turns the product into a sum.
* **Out-of-law extent** -- the capacity law ``capacity = extent/σ`` (FC-4)
  has a measured sweet spot near **10σ per axis** for multi-dimensional
  hidden charts (FC-4's 2-D result, re-confirmed by the FC-7a3/a4 bracket:
  5σ too narrow, 20σ collapsed).  Data-pinned input charts ran best near
  20--30σ per axis, and 1-D hidden charts are population-dependent
  (60--80σ at H = 64), so the window reported here is advisory for the
  multi-dimensional hidden case, not a hard gate.

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

__all__ = ["AxisSurvey", "ChartSurvey", "survey_chart"]

#: Default neuron spacing (in σ units) beyond which an axis is effectively
#: discrete: the kernel factor across one gap is exp(-spacing²/2) ≤ 4e-6.
QUASI_DISCRETE_SPACING = 5.0

#: Default advisory extent window (in σ units per axis) for multi-dimensional
#: hidden charts: the measured peak sits near 10σ/axis, bracketed by a
#: too-narrow 5σ and a coverage-collapsing 20σ (FC-4 appendix J; FC-7a3/a4).
LAWFUL_AXIS_EXTENT = (5.0, 20.0)

#: Default atoms-per-resolvable-cell ratio below which the chart is flagged
#: as coverage-starved (fc7h's lawful-joint collapse; the 20σ box collapse).
COVERAGE_FLOOR = 0.5


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
    """

    sigma: float
    population: int
    axes: tuple[AxisSurvey, ...]
    cells: float
    coverage: float | None
    notes: tuple[str, ...]

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
    if coverage is not None and coverage < coverage_floor:
        notes.append(
            f"coverage {coverage:.2f} atoms/cell < {coverage_floor} — "
            "coverage-starved; the capacity law is only affordable together "
            "with chart factorization (volume must be a sum, not a product)"
        )
    if len(axes) > 1:
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
    )
