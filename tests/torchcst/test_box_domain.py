"""Box continuous-coordinate domain contracts."""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import Box, ParameterRole


def test_box_sample_retract_and_identity_gradient() -> None:
    box = Box(-2.0, 3.0, 2)
    sample = box.sample(256, torch.Generator().manual_seed(9))

    assert box.parameter_role() is ParameterRole.PARAMETER
    assert sample.shape == (256, 2)
    assert bool(((sample >= -2.0) & (sample <= 3.0)).all())
    box.validate_birth(sample)
    assert box.lineage_key(sample) is None

    raw = torch.tensor([[-4.0, 1.5], [9.0, -3.0]])
    torch.testing.assert_close(
        box.retract(raw), torch.tensor([[-2.0, 1.5], [3.0, -2.0]])
    )
    grad = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    assert box.project_grad(box.retract(raw), grad) is grad


def test_box_project_state_is_a_no_op_without_a_direction_gauge() -> None:
    box = Box(0.0, 1.0, 2)
    coords = torch.tensor([[0.0, 1.0]])
    momentum = torch.tensor([[4.0, -3.0]])
    second_moment = torch.tensor([[2.0, 5.0]])
    state = {"momentum_buffer": momentum, "exp_avg_sq": second_moment}
    expected = {name: value.clone() for name, value in state.items()}

    box.project_state(coords, state)

    for name, value in state.items():
        assert torch.equal(value, expected[name])


def test_a_cube_keeps_the_scalar_bound_attributes_and_the_old_sample_law() -> None:
    """The per-axis form must not perturb the cube it generalizes: same
    scalar ``lo``/``hi``/``bounds``, same ``width``, and -- because every
    downstream draw is seeded off the same generator -- the same samples
    element for element."""
    cube = Box(-2.0, 3.0, 2)
    per_axis = Box([-2.0, -2.0], [3.0, 3.0], 2)

    for box in (cube, per_axis):
        assert box.lo == -2.0 and box.hi == 3.0
        assert box.bounds == (-2.0, 3.0)
        assert box.uniform is True
        assert box.width == 5.0
        assert box.widths == (5.0, 5.0)

    a = cube.sample(64, torch.Generator().manual_seed(3))
    b = per_axis.sample(64, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_an_anisotropic_box_samples_validates_and_retracts_per_axis() -> None:
    """An isotropic chart is not a cube (a 64-point axis and a 3-point axis
    at equal spacing span 63h and 2h), so every coordinate law the atoms and
    the candidate pool go through has to be per-axis or the geometry fix is
    defeated by the sampling law."""
    box = Box([0.0, 0.4, 0.4], [1.0, 0.6, 0.6], 3)

    assert box.uniform is False
    assert box.lo == (0.0, 0.4, 0.4)
    assert box.widths == (1.0, pytest.approx(0.2), pytest.approx(0.2))
    # volume ** (1/dim), i.e. the edge of the equal-volume cube.
    assert box.width == pytest.approx((1.0 * 0.2 * 0.2) ** (1.0 / 3.0))

    sample = box.sample(4096, torch.Generator().manual_seed(11))
    assert float(sample[:, 0].max()) > 0.9
    assert bool((sample[:, 1] >= 0.4).all()) and bool((sample[:, 1] <= 0.6).all())
    assert bool((sample[:, 2] >= 0.4).all()) and bool((sample[:, 2] <= 0.6).all())
    box.validate_birth(sample)

    outside = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 0.5, 0.5]])
    torch.testing.assert_close(
        box.retract(outside), torch.tensor([[0.0, 0.4, 0.6], [1.0, 0.5, 0.5]])
    )
    with pytest.raises(ValueError, match="out of bounds"):
        box.validate_birth(outside)


def test_box_rejects_a_per_axis_bound_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="dim entries"):
        Box([0.0, 0.0], 1.0, 3)
