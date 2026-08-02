"""``Box(..., retract_in_training=...)``: opt out of the training-time clamp.

``SynapseStore.retract_coordinates`` runs on every structural event and
clamps live coordinates back into the box, which measurably constrains
learning and not just initialization (see ``README.md``, "Choosing the
coordinate domain"). ``retract_in_training=False`` turns off *only* that
clamp; sampling birth candidates and validating structural writes must keep
enforcing the box regardless, so a candidate can never be proposed or
committed outside the domain it was drawn to represent.

Default ``True`` must reproduce today's unconditional clamp exactly.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import Box, RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def _store(box_in: Box, box_out: Box) -> SynapseStore:
    spec = RepresentationSpec(
        domain_in=box_in,
        domain_out=box_out,
        kernel_in="gaussian",
        kernel_out="gaussian",
        atom_cost=box_in.dim + box_out.dim + 1,
        retirement="gate_only",
    )
    store = SynapseStore("layer", 1, 1, capacity=4, spec=spec, dtype=torch.float64)
    store.apply(
        [
            SynapseBirth(
                "layer",
                torch.tensor([[0.0], [0.5]], dtype=torch.float64),
                torch.tensor([[0.0], [0.5]], dtype=torch.float64),
                torch.tensor([0.1, 0.1], dtype=torch.float64),
                torch.arange(2, dtype=torch.int64),
            )
        ]
    )
    return store


def test_default_is_true_and_matches_todays_unconditional_clamp() -> None:
    box = Box(-1.0, 1.0, 1)

    assert box.retract_in_training is True

    store = _store(box, Box(-1.0, 1.0, 1))
    with torch.no_grad():
        store.s.index_copy_(
            0, store._slots.live_slots, torch.tensor([[-4.0], [9.0]], dtype=torch.float64)
        )

    store.retract_coordinates()

    live = store.s.index_select(0, store._slots.live_slots)
    torch.testing.assert_close(live, torch.tensor([[-1.0], [1.0]], dtype=torch.float64))


def test_retract_in_training_false_leaves_live_coordinates_unclamped() -> None:
    box = Box(-1.0, 1.0, 1, retract_in_training=False)
    store = _store(box, Box(-1.0, 1.0, 1))
    out_of_bounds = torch.tensor([[-4.0], [9.0]], dtype=torch.float64)
    with torch.no_grad():
        store.s.index_copy_(0, store._slots.live_slots, out_of_bounds)

    store.retract_coordinates()

    live = store.s.index_select(0, store._slots.live_slots)
    torch.testing.assert_close(live, out_of_bounds)


def test_retract_in_training_false_still_enforces_sampling_and_validation() -> None:
    box = Box(-1.0, 1.0, 1, retract_in_training=False)

    sample = box.sample(256, torch.Generator().manual_seed(3))
    assert bool(((sample >= -1.0) & (sample <= 1.0)).all())
    box.validate_birth(sample)

    with pytest.raises(ValueError, match="out of bounds"):
        box.validate_birth(torch.tensor([[5.0]], dtype=torch.float64))


def test_retract_in_training_false_does_not_change_direct_retract_calls() -> None:
    """The flag is a store-call-site opt-out, not a change to ``retract`` itself --
    candidate sampling (``instruments/_candidate_sampling.py``) calls
    ``Box.retract`` directly to clip proposals into the domain, and that path
    must keep clamping even when the training-time projection is off."""
    box = Box(-1.0, 1.0, 1, retract_in_training=False)

    clamped = box.retract(torch.tensor([[-4.0], [9.0]], dtype=torch.float64))

    torch.testing.assert_close(clamped, torch.tensor([[-1.0], [1.0]], dtype=torch.float64))


def test_retract_in_training_rejects_non_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        Box(-1.0, 1.0, 1, retract_in_training=1)
