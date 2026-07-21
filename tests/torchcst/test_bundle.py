"""ProposalBundle accounting and cross-store atomicity contracts."""

from __future__ import annotations

import torch

from torchcst.compute import EntryLinear, NeuronGatedLinear, RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import EvenBudgetAllocator, LC, ProposalBundle
from torchcst.representation import RepresentationSpec
from torchcst.storage import (
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseStore,
)


def birth(site: str, count: int, *, target: int = 0) -> SynapseBirth:
    return SynapseBirth(
        site,
        torch.arange(count, dtype=torch.int64).reshape(count, 1),
        torch.full((count, 1), target, dtype=torch.int64),
        torch.ones(count),
        torch.arange(count, dtype=torch.int64),
    )


def test_allocator_accounts_birth_rows_and_never_splits_a_bundle() -> None:
    first = ProposalBundle(
        "first", (NeuronUngate("n", torch.tensor([0])), birth("s", 2))
    )
    second = ProposalBundle(
        "second", (NeuronUngate("n", torch.tensor([1])), birth("s", 2))
    )

    accepted = EvenBudgetAllocator().allocate_bundles(3, (first, second))

    assert accepted == (first,)
    assert accepted[0].ops == first.ops


def test_invalid_bundle_drops_only_it_and_never_applies_its_ungate() -> None:
    synapses = SynapseStore("edge", 1, 1, 4)
    neurons = NeuronStore("hidden", 3)
    engine = StructuralEngine({"edge": synapses, "hidden": neurons}, LC(birth_budget=0))
    invalid_birth = SynapseBirth(
        "edge",
        torch.zeros(1, 2, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.ones(1),
        torch.zeros(1, dtype=torch.int64),
    )
    bad = ProposalBundle(
        "bad",
        (NeuronUngate("hidden", torch.tensor([0])), invalid_birth),
    )
    good = ProposalBundle("good", (NeuronUngate("hidden", torch.tensor([1])),))

    applied = engine.apply_proposals(
        [bad, good, NeuronUngate("hidden", torch.tensor([2]))]
    )

    assert neurons.state.tolist() == [0, 1, 1]
    assert neurons.gate_vector()[0].item() == 0.0
    assert [op.ids.item() for op in applied] == [1, 2]


def test_entry_retirement_cascades_both_endpoint_sides_in_one_plan() -> None:
    synapses = SynapseStore(
        "edge",
        1,
        1,
        4,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    synapses.apply(
        [
            SynapseBirth(
                "edge",
                torch.tensor([[0], [1], [0]], dtype=torch.int64),
                torch.tensor([[0], [0], [1]], dtype=torch.int64),
                torch.ones(3),
                torch.arange(3, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore("inputs", 2, initial_live=2)
    outputs = NeuronStore("outputs", 2, initial_live=2)
    module = NeuronGatedLinear(EntryLinear(synapses, 2, 2), inputs, outputs)
    engine = StructuralEngine(
        {"edge": synapses, "inputs": inputs, "outputs": outputs},
        LC(birth_budget=0),
        modules={"edge": module},
    )

    applied = engine.apply_proposals([NeuronRetire("inputs", torch.tensor([0]))])

    assert [type(op).__name__ for op in applied] == [
        "NeuronRetire",
        "SynapseDeath",
    ]
    assert synapses.view().s[:, 0].tolist() == [1]


def test_rank_one_retirement_is_gate_only_until_projection_op_exists() -> None:
    synapses = SynapseStore("rank", 2, 2, 1, spec=RepresentationSpec.rank_one(2, 2))
    synapses.apply(
        [
            SynapseBirth(
                "rank",
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[0.0, 1.0]]),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore("inputs", 2, initial_live=2)
    outputs = NeuronStore("outputs", 2, initial_live=2)
    module = NeuronGatedLinear(RankOneLinear(synapses, 2, 2), inputs, outputs)
    engine = StructuralEngine(
        {"rank": synapses, "inputs": inputs, "outputs": outputs},
        LC(birth_budget=0),
        modules={"rank": module},
    )

    applied = engine.apply_proposals([NeuronRetire("outputs", torch.tensor([1]))])

    assert [type(op).__name__ for op in applied] == ["NeuronRetire"]
    assert synapses.view().ids.numel() == 1
