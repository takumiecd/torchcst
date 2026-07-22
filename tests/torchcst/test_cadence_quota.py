"""Timing, logical quotas, and physical slot placement stay independent."""

from __future__ import annotations

import torch

from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ActionSpec,
    BudgetRequest,
    CallableQuota,
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    Phase,
    Policy,
    StructuralQuota,
    UniformEntryBirth,
)
from torchcst.policy.contract import Clock
from torchcst.representation import RepresentationSpec
from torchcst.storage import (
    NeuronRetire,
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def test_cadence_signal_contains_no_structural_supply() -> None:
    cadence = PeriodicCadence(event_interval=2, observe_window=1)
    signal = cadence.event(Clock(update_step=2, event_index=1))
    assert signal is not None
    assert signal.phase is Phase.GROW
    assert not hasattr(signal, "birth_budget")


def test_quota_can_be_constant_or_clock_annealed() -> None:
    fixed = ConstantQuota(StructuralQuota(synapse_birth=8, neuron_birth=2))
    assert fixed.at(Clock(1, 1), Phase.GROW).synapse_birth == 8
    assert fixed.at(Clock(2, 2), Phase.FROZEN) == StructuralQuota.zero()

    annealed = CallableQuota(
        lambda clock, phase: StructuralQuota(
            synapse_birth=max(0, 10 - clock.update_step)
        )
    )
    assert annealed.at(Clock(3, 1), Phase.GROW).synapse_birth == 7


def test_distributor_allocates_logical_quota_not_storage_slots() -> None:
    distributor = EvenBudgetDistributor()
    requests = (
        BudgetRequest("left", 0, 0),
        BudgetRequest("right", 0, 0),
    )
    assert distributor.allocate(5, requests) == (3, 2)
    assert not hasattr(distributor, "slots_of")
    assert not hasattr(distributor, "prepare")


def test_engine_runs_policy_with_separate_cadence_and_quota() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=2,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=1)),
        actions=(
            ActionSpec.synapse_prune(MagnitudeCourt(0.0)),
            ActionSpec.synapse_birth(
                UniformEntryBirth(bounds_in=1, bounds_out=2)
            ),
        ),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine({store.site: store}, policy, seed=4).step()
    assert sum(isinstance(operation, SynapseBirth) for operation in operations) == 1
    assert store.live_ids().numel() == 1


def test_replacement_reuses_slot_under_new_policy_api() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=1,
        max_capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int64),
                torch.ones(1),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    old_id = store.live_ids().clone()
    old_slot = store._slots.slots_of(old_id)
    replacement = SynapseBirth(
        store.site,
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int64),
        torch.zeros(1),
        torch.tensor([1], dtype=torch.int64),
    )
    store.apply((SynapseDeath(store.site, old_id), replacement))
    assert store.capacity == 1
    assert torch.equal(store._slots.slots_of(store.live_ids()), old_slot)
    assert not torch.equal(store.live_ids(), old_id)


def test_synapse_prune_quota_caps_court_decision() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=4,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=4),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.zeros((4, 1), dtype=torch.int64),
                torch.arange(4, dtype=torch.int64)[:, None],
                torch.arange(1, 5, dtype=torch.float32),
                torch.arange(4, dtype=torch.int64),
            )
        ]
    )
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_prune=2)),
        actions=(ActionSpec.synapse_prune(MagnitudeCourt(1.0)),),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine({store.site: store}, policy).step()

    deaths = [
        operation
        for operation in operations
        if isinstance(operation, SynapseDeath)
    ]
    assert sum(operation.ids.numel() for operation in deaths) == 2
    assert store.live_ids().numel() == 2


def test_unbounded_prune_quota_preserves_court_decision() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=2,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.zeros((2, 1), dtype=torch.int64),
                torch.arange(2, dtype=torch.int64)[:, None],
                torch.ones(2),
                torch.arange(2, dtype=torch.int64),
            )
        ]
    )
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota()),
        actions=(ActionSpec.synapse_prune(MagnitudeCourt(1.0)),),
        distributor=EvenBudgetDistributor(),
    )

    StructuralEngine({store.site: store}, policy).step()

    assert store.live_ids().numel() == 0


def test_neuron_prune_quota_caps_court_decision() -> None:
    synapses = SynapseStore(
        "edge",
        1,
        1,
        capacity=0,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=3),
    )
    neurons = NeuronStore("outputs", 3, initial_live=3)
    with torch.no_grad():
        neurons.gate.copy_(torch.tensor([0.1, 0.2, 0.3]))
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(neuron_prune=1)),
        actions=(ActionSpec.neuron_prune(MagnitudeCourt(1.0)),),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine(
        {synapses.site: synapses, neurons.site: neurons}, policy
    ).step()

    retirements = [
        operation
        for operation in operations
        if isinstance(operation, NeuronRetire)
    ]
    assert sum(operation.ids.numel() for operation in retirements) == 1
    assert neurons.live_ids().numel() == 2
