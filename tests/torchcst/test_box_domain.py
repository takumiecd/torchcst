"""Box continuous-coordinate domain contracts."""

from __future__ import annotations

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
