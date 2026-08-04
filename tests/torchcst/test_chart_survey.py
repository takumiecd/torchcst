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


def test_wide_box_is_coverage_starved():
    # ~30 sigma per axis with the same 64 atoms: ~850 cells, coverage < 0.1
    # -- the fc7h / wide-box collapse regime, outside the advisory window.
    generator = torch.Generator().manual_seed(0)
    coords = torch.rand(64, 2, generator=generator) * 3.0
    survey = survey_chart(coords, 0.1, atoms=64)
    assert survey.coverage is not None and survey.coverage < 0.5
    assert any("coverage-starved" in note for note in survey.notes)
    assert any("advisory window" in note for note in survey.notes)


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
    assert proposal.dim == 2
    assert proposal.box.bounds == (0.0, pytest.approx(1.0))
    assert proposal.cells == pytest.approx(100.0)
    assert proposal.recommended_atoms == 100
    assert proposal.notes == ()

    rng = torch.Generator().manual_seed(0)
    coords = proposal.box.sample(64, rng)
    survey = survey_chart(coords, 0.1, atoms=proposal.recommended_atoms)
    assert survey.notes == ()


def test_propose_chart_scales_dimension_with_population():
    from torchcst.representation import propose_chart

    assert propose_chart(8, 0.1).dim == 1
    assert propose_chart(64, 0.1).dim == 2
    assert propose_chart(500, 0.1).dim == 3


def test_propose_chart_reports_forced_disagreements():
    from torchcst.representation import propose_chart

    forced = propose_chart(64, 0.1, dim=1)
    assert any("resolvable cells" in note for note in forced.notes)
    starved = propose_chart(64, 0.1, atoms=10)
    assert any("coverage-starved" in note for note in starved.notes)


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
        propose_chart(64, 0.1, axis_extent=1.0)


def test_thresholds_are_configurable():
    from torchcst.representation import propose_chart

    # Wider lawful target: same population, differently sized lawful box.
    wide = propose_chart(64, 0.1, axis_extent=20.0)
    assert wide.dim == 2
    assert wide.box.bounds == (0.0, pytest.approx(2.0))
    assert wide.cells == pytest.approx(400.0)

    # Raising the quasi-discrete threshold unflags the 5-sigma conv taps.
    input_mu, _ = conv2d_neuron_coordinates(16, 32, 3)
    relaxed = survey_chart(input_mu, 0.1, quasi_discrete_spacing=6.0)
    assert not any(axis.quasi_discrete for axis in relaxed.axes)
