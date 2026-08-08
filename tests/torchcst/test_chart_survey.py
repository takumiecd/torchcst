"""Contract tests for the chart operating-envelope survey.

Each test pins one measured regime from the 2026-08 conv arc (provenance in
``survey.py``'s module docstring): quasi-discrete spatial axes, over-dense
1-D channel lines, coverage starvation in wide/joint boxes, and the lawful
2-D window where everything worked.
"""

import math

import pytest
import torch

from torchcst.compute import conv2d_neuron_coordinates
from torchcst.representation import survey_chart


def _grid_chart(side: int) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)


def test_dense_image_chart_is_continuous_on_every_axis():
    # 28x28 pixels on [0,1]^2 at sigma=0.1: spacing 0.37 sigma -- the regime
    # where linear CST charts measurably work.
    survey = survey_chart(_grid_chart(28), 0.1)
    assert survey.dim == 2
    for axis in survey.axes:
        assert not axis.quasi_discrete
        assert axis.spacing_over_sigma == pytest.approx(1.0 / 27.0 / 0.1, rel=1e-6)
    assert not any("quasi-discrete" in note for note in survey.notes)


def test_conv_tap_axes_are_flagged_quasi_discrete():
    # The FC-6a conv chart at sigma=0.1: 3 taps per spatial axis at 5 sigma
    # spacing.  Both spatial axes must be flagged; the channel axis (16
    # neurons, 0.67 sigma spacing) must not be.
    input_mu, _ = conv2d_neuron_coordinates(16, 32, 3)
    survey = survey_chart(input_mu, 0.1)
    assert not survey.axes[0].quasi_discrete            # channel axis
    assert survey.axes[1].quasi_discrete                # tap y
    assert survey.axes[2].quasi_discrete                # tap x
    flagged = [note for note in survey.notes if "quasi-discrete" in note]
    assert len(flagged) == 2


def test_overdense_channel_line_flags_population():
    # 64 channels on a 10-sigma 1-D line: at most ~10 resolvable cells, so
    # the population cannot be told apart (FC-7a's sigma_c=1.0 regime).
    coords = torch.linspace(0.0, 1.0, 64).unsqueeze(1)
    survey = survey_chart(coords, 0.1)
    assert survey.cells == pytest.approx(10.0)
    assert any("resolvable cells" in note for note in survey.notes)


def test_lawful_2d_box_with_covering_atoms_is_clean():
    # The FC-7a4 winner: 64 channels in a [0,1]^2 box at sigma=0.1 with 64
    # atoms -- 100 cells, coverage 0.64, population within capacity.
    generator = torch.Generator().manual_seed(0)
    coords = torch.rand(64, 2, generator=generator)
    survey = survey_chart(coords, 0.1, atoms=64)
    assert survey.coverage is not None and survey.coverage > 0.5
    assert survey.population <= survey.cells
    assert survey.notes == ()


def test_wide_but_correctly_spaced_box_is_not_flagged():
    # OVERTURNS the pre-DIM-K2 contract, which flagged this chart as
    # coverage-starved.  ~30 sigma per axis with 64 neurons and 64 atoms is
    # coverage 0.074, thirteen times under the old floor -- but its neuron
    # spacing is 3.68 sigma, inside dim 2's measured 0.7-4.0 window, and
    # DIM-K2 trained 2-D charts across that whole band (spacing 4.0 scored
    # 0.4027 against the 0.4215 peak).  Coverage was reading a symptom.
    generator = torch.Generator().manual_seed(0)
    coords = torch.rand(64, 2, generator=generator) * 3.0
    survey = survey_chart(coords, 0.1, atoms=64)
    assert survey.coverage is not None and survey.coverage < 0.5
    assert survey.lattice_spacing_over_sigma == pytest.approx(3.68, abs=0.02)
    assert survey.in_spacing_window
    assert survey.notes == ()


def test_high_dimensional_joint_chart_is_flagged_by_spacing():
    # The failure the coverage floor was actually fitted to: a 6-D joint chart
    # whose spacing (7.18 sigma) has left dim 6's narrow 2.0-2.8 window.  The
    # spacing note is the diagnosis; the coverage note now rides along with it
    # instead of firing on its own.
    coords = torch.rand(64, 6, generator=torch.Generator().manual_seed(1)) * 1.5
    survey = survey_chart(coords, 0.1, atoms=64)
    assert not survey.in_spacing_window
    assert survey.lattice_spacing_over_sigma > 2.8
    assert any("neuron spacing" in note and "above" in note for note in survey.notes)
    assert any("coverage-starved as well" in note for note in survey.notes)


def test_spacing_window_narrows_with_dimension():
    from torchcst.representation.survey import RECOMMENDED_SPACING, spacing_window

    # The measured shape: the band narrows and its centre rises with dimension.
    assert spacing_window(1) == (0.7, 4.0)
    assert spacing_window(6) == (2.0, 2.8)
    for dim in range(1, 7):
        low, high = spacing_window(dim)
        assert low <= RECOMMENDED_SPACING <= high      # the universal spacing
    # Unmeasured dimensions read conservatively: 5 intersects 4 and 6, and
    # anything past the largest measured dimension inherits its band.
    assert spacing_window(5) == (2.0, 2.8)
    assert spacing_window(8) == spacing_window(6)


def test_degenerate_axis_reported():
    coords = torch.zeros(8, 1)
    survey = survey_chart(coords, 0.1)
    assert survey.axes[0].unique_positions == 1
    assert math.isinf(survey.axes[0].spacing_over_sigma)
    assert any("degenerate" in note for note in survey.notes)


def test_input_validation():
    with pytest.raises(TypeError):
        survey_chart([[0.0, 1.0]], 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        survey_chart(torch.zeros(0, 2), 0.1)
    with pytest.raises(ValueError):
        survey_chart(torch.zeros(4, 2, 2), 0.1)
    with pytest.raises(ValueError):
        survey_chart(torch.zeros(4, 2), 0.0)
    with pytest.raises(ValueError):
        survey_chart(torch.full((4, 2), float("nan")), 0.1)
    with pytest.raises(TypeError):
        survey_chart(torch.zeros(4, 2), 0.1, atoms=True)
    with pytest.raises(ValueError):
        survey_chart(torch.zeros(4, 2), 0.1, atoms=0)


def test_propose_chart_round_trips_through_its_own_survey():
    # The proposal is executable: sampling neurons and atoms from the
    # proposed box must produce a chart the survey finds nothing to flag.
    from torchcst.representation import propose_chart

    proposal = propose_chart(64, 0.1)
    assert proposal.dim == 3                       # DEFAULT_DIM, DIM-K2's peak
    assert proposal.spacing == pytest.approx(2.0)  # RECOMMENDED_SPACING
    # extent = spacing * population**(1/dim) = 2 * 4 = 8 sigma per axis
    assert proposal.box.bounds == (0.0, pytest.approx(0.8))
    assert proposal.cells == pytest.approx(512.0)
    assert proposal.recommended_atoms == 64        # one atom per neuron
    assert proposal.notes == ()

    rng = torch.Generator().manual_seed(0)
    coords = proposal.box.sample(64, rng)
    survey = survey_chart(coords, 0.1, atoms=proposal.recommended_atoms)
    assert survey.in_spacing_window
    assert survey.notes == ()


def test_propose_chart_reproduces_the_measured_optimum():
    # DIM-K2's best arm over 105 runs: 1024 neurons at dim 3 with 20.2 sigma
    # per axis and 1024 atoms.  The proposal must land on it unprompted --
    # that is the whole claim of "lawful by construction".
    from torchcst.representation import propose_chart

    proposal = propose_chart(1024, 0.1)
    assert proposal.dim == 3
    assert proposal.box.hi / 0.1 == pytest.approx(20.16, abs=0.01)
    assert proposal.recommended_atoms == 1024
    assert proposal.notes == ()


def test_propose_chart_holds_dimension_and_scales_extent_with_population():
    # REPLACES the pre-DIM-K2 contract, where dim grew with population to keep
    # cells >= population.  Distinguishability is a spacing property, so every
    # dimension can hold any population; what scales with population is the
    # extent needed to keep spacing fixed.
    from torchcst.representation import propose_chart

    for population in (8, 64, 500, 4096):
        assert propose_chart(population, 0.1).dim == 3
        assert propose_chart(population, 0.1).spacing == pytest.approx(2.0)
    # extent ratio between 8 and 4096 neurons is (4096/8)**(1/3) = 8
    small = propose_chart(8, 0.1).box.hi
    large = propose_chart(4096, 0.1).box.hi
    assert large / small == pytest.approx(8.0)


def test_propose_chart_reports_forced_disagreements():
    from torchcst.representation import propose_chart

    # A forced dim=1 is no longer a disagreement: DIM-K2 trained dim 1 at the
    # recommended spacing (0.4127 against the 0.4293 peak).
    assert propose_chart(64, 0.1, dim=1).notes == ()
    outside = propose_chart(64, 0.1, spacing=0.3)
    assert any("below the measured" in note for note in outside.notes)
    thin = propose_chart(64, 0.1, atoms=10)
    assert any("fewer atoms than neurons" in note for note in thin.notes)


def test_propose_chart_validation():
    from torchcst.representation import propose_chart

    with pytest.raises(ValueError):
        propose_chart(0, 0.1)
    with pytest.raises(ValueError):
        propose_chart(64, 0.0)
    with pytest.raises(TypeError):
        propose_chart(64, 0.1, dim=True)
    with pytest.raises(ValueError):
        propose_chart(64, 0.1, atoms=-1)
    with pytest.raises(ValueError):
        propose_chart(64, 0.1, spacing=0.0)


def test_thresholds_are_configurable():
    from torchcst.representation import propose_chart

    # Sitting elsewhere in the measured band: same population and dimension,
    # a proportionally larger box.  2.8 sigma is dim 3's upper bound, so this
    # is still lawful and must not be flagged.
    wide = propose_chart(64, 0.1, spacing=2.8)
    assert wide.dim == 3
    assert wide.box.bounds == (0.0, pytest.approx(1.12))
    assert wide.spacing == pytest.approx(2.8)
    assert wide.notes == ()

    # Raising the quasi-discrete threshold unflags the 5-sigma conv taps.
    input_mu, _ = conv2d_neuron_coordinates(16, 32, 3)
    relaxed = survey_chart(input_mu, 0.1, quasi_discrete_spacing=6.0)
    assert not any(axis.quasi_discrete for axis in relaxed.axes)
