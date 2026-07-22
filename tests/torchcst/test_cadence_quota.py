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
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


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
