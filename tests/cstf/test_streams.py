"""Named RNG stream isolation and restoration tests."""

from __future__ import annotations

import pytest
import torch

from cstf.lab import RngStreams


def draw(generator: torch.Generator, count: int = 12) -> torch.Tensor:
    return torch.rand(count, generator=generator)


def test_named_streams_are_reproducible_distinct_and_stable_instances() -> None:
    left = RngStreams(314159)
    right = RngStreams(314159)

    assert left.get("data") is left.get("data")
    assert torch.equal(draw(left.get("data")), draw(right.get("data")))

    fresh = RngStreams(314159)
    assert not torch.equal(draw(fresh.get("data")), draw(fresh.get("proposal")))


def test_consuming_proposal_never_changes_data_stream() -> None:
    consumed = RngStreams(7)
    untouched = RngStreams(7)

    draw(consumed.get("proposal"), 1000)

    assert torch.equal(draw(consumed.get("data")), draw(untouched.get("data")))


def test_new_stream_requires_registration_and_does_not_perturb_existing_ones() -> None:
    streams = RngStreams(81)
    reference = RngStreams(81)

    with pytest.raises(KeyError, match="register"):
        streams.get("proposals")
    custom = streams.register("calibration")

    assert custom is streams.get("calibration")
    assert torch.equal(draw(streams.get("data")), draw(reference.get("data")))


def test_state_dict_restores_every_stream_and_custom_registry() -> None:
    streams = RngStreams(12)
    streams.register("trial")
    draw(streams.get("data"), 3)
    draw(streams.get("trial"), 5)
    snapshot = streams.state_dict()
    expected_data = draw(streams.get("data"))
    expected_trial = draw(streams.get("trial"))

    streams.register("temporary")
    streams.load_state_dict(snapshot)

    assert torch.equal(draw(streams.get("data")), expected_data)
    assert torch.equal(draw(streams.get("trial")), expected_trial)
    with pytest.raises(KeyError):
        streams.get("temporary")


def test_load_state_dict_restores_root_seed_into_a_fresh_owner() -> None:
    source = RngStreams(-9)
    draw(source.get("diag"), 4)
    state = source.state_dict()
    expected = draw(source.get("diag"))
    restored = RngStreams(999)

    restored.load_state_dict(state)

    assert restored.root_seed == -9
    assert torch.equal(draw(restored.get("diag")), expected)
