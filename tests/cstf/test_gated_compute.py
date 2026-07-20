"""Numerical gate masking for entry and rank-one compute paths."""

from __future__ import annotations

import pytest
import torch

from cstf.compute import EntryLinear, RankOneLinear
from cstf.representation import RepresentationSpec
from cstf.storage import (
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseStore,
    commit_all,
    prepare_all,
)


def entry_module():
    store = SynapseStore(
        "entry",
        1,
        1,
        4,
        spec=RepresentationSpec.entry(bounds_in=4, bounds_out=4),
    )
    coordinates = torch.arange(4, dtype=torch.int64).reshape(4, 1)
    store.apply(
        [
            SynapseBirth(
                "entry", coordinates, coordinates, torch.ones(4), torch.arange(4)
            )
        ]
    )
    inputs = NeuronStore("inputs", 4, initial_live=2)
    outputs = NeuronStore("outputs", 4, initial_live=2)
    inputs.apply([NeuronRetire("inputs", torch.tensor([1]))])
    outputs.apply([NeuronRetire("outputs", torch.tensor([1]))])
    return EntryLinear(store, 4, 4, inputs, outputs), inputs, outputs


def rank_one_module():
    store = SynapseStore(
        "rank", 4, 4, 4, spec=RepresentationSpec.rank_one(4, 4)
    )
    factors = torch.eye(4)
    store.apply(
        [
            SynapseBirth(
                "rank", factors, factors, torch.ones(4), torch.arange(4)
            )
        ]
    )
    inputs = NeuronStore("inputs", 4, initial_live=2)
    outputs = NeuronStore("outputs", 4, initial_live=2)
    inputs.apply([NeuronRetire("inputs", torch.tensor([1]))])
    outputs.apply([NeuronRetire("outputs", torch.tensor([1]))])
    return RankOneLinear(store, 4, 4, inputs, outputs), inputs, outputs


@pytest.mark.parametrize("factory", [entry_module, rank_one_module])
def test_dormant_and_retired_contributions_are_zero_then_ungate_appears(factory) -> None:
    module, inputs, outputs = factory()
    x = torch.tensor([[2.0, 3.0, 5.0, 7.0]])

    torch.testing.assert_close(module(x), torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    tickets = prepare_all(
        [
            (inputs, [NeuronUngate("inputs", torch.tensor([2]), 2.0)]),
            (outputs, [NeuronUngate("outputs", torch.tensor([2]), 3.0)]),
        ]
    )
    commit_all(tickets)

    torch.testing.assert_close(module(x), torch.tensor([[2.0, 0.0, 30.0, 0.0]]))


@pytest.mark.parametrize("factory", [entry_module, rank_one_module])
def test_gate_gradient_has_exact_zeros_outside_live_state(factory) -> None:
    module, inputs, outputs = factory()
    inputs.apply([NeuronUngate("inputs", torch.tensor([2]), 2.0)])
    outputs.apply([NeuronUngate("outputs", torch.tensor([2]), 3.0)])
    module(torch.ones(1, 4)).sum().backward()

    assert inputs.gate.grad is not None and outputs.gate.grad is not None
    assert torch.count_nonzero(inputs.gate.grad[[0, 2]]) == 2
    assert torch.count_nonzero(outputs.gate.grad[[0, 2]]) == 2
    assert torch.equal(inputs.gate.grad[[1, 3]], torch.zeros(2))
    assert torch.equal(outputs.gate.grad[[1, 3]], torch.zeros(2))
