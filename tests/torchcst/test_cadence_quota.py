"""Timing, logical quotas, and physical slot placement stay independent."""

from __future__ import annotations

import torch

from torchcst.compute import EntryLinear, NeuronGatedLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    BudgetRequest,
    CallableQuota,
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    NeuronLifecycle,
    PeriodicCadence,
    Phase,
    QuotaRegime,
    StructuralQuota,
    SynapseLifecycle,
    UniformEntryBirth,
)
from torchcst.policy.contract import Clock
from torchcst.representation import RepresentationSpec
from torchcst.storage import (
    NeuronUngate,
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


def test_engine_runs_tree_with_separate_cadence_and_quota() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=2,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    method = SynapseLifecycle(
        birth_factory=lambda lam: UniformEntryBirth(bounds_in=1, bounds_out=2),
        prune_factory=lambda: MagnitudeCourt(0.0),
        priceable=False,
        label="test",
    )
    root = QuotaRegime(
        budget=1,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine({store.site: store}, root, seed=4).step()
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
    method = SynapseLifecycle(
        birth_factory=lambda lam: UniformEntryBirth(bounds_in=1, bounds_out=4),
        prune_factory=lambda: MagnitudeCourt(1.0),
        priceable=False,
        label="test",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_prune=2)),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine({store.site: store}, root).step()

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
    method = SynapseLifecycle(
        birth_factory=lambda lam: UniformEntryBirth(bounds_in=1, bounds_out=2),
        prune_factory=lambda: MagnitudeCourt(1.0),
        priceable=False,
        label="test",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota()),
        distributor=EvenBudgetDistributor(),
    )

    StructuralEngine({store.site: store}, root).step()

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
    # The interface seat (NeuronLifecycle) only binds to a neuron store that
    # is some synapse module's out_neurons (policy/tree.py's _bind_root); a
    # bare NeuronStore with no compute module never becomes an InterfaceChild
    # and so would never have its court consulted at all.
    module = NeuronGatedLinear(EntryLinear(synapses, 1, 3), out_neurons=neurons)
    method = SynapseLifecycle(
        birth_factory=lambda lam: UniformEntryBirth(bounds_in=1, bounds_out=3),
        priceable=False,
        label="test",
    )
    interface = NeuronLifecycle(
        retention_factory=lambda: MagnitudeCourt(1.0),
        label="test-interface",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(neuron_prune=1)),
        distributor=EvenBudgetDistributor(),
        interface=interface,
    )

    operations = StructuralEngine(
        {synapses.site: synapses, neurons.site: neurons},
        root,
        modules={synapses.site: module},
    ).step()

    retirements = [
        operation
        for operation in operations
        if isinstance(operation, NeuronRetire)
    ]
    assert sum(operation.ids.numel() for operation in retirements) == 1
    assert neurons.live_ids().numel() == 2


def _growing_world(*, quota=None):
    """One CST map + terminal boundary whose output chart has dormant rows."""
    from torchcst.compute import CSTBoundary, CSTLinear
    from torchcst.policy import cSFW, gamma_ungate
    from torchcst.representation import GaussianKernel

    store = SynapseStore(
        "layer",
        1,
        1,
        32,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    grid = torch.linspace(0.1, 0.9, 6, dtype=torch.float64)[:, None]
    store.apply(
        [
            SynapseBirth(
                "layer",
                grid.clone(),
                grid.flip(0).clone(),
                torch.full((6,), 0.05, dtype=torch.float64),
                torch.arange(6),
            )
        ]
    )
    mu_in = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)[:, None]
    mu_out = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)[:, None]
    inputs = NeuronStore("in", 4, mu=mu_in, initial_live=4, dtype=torch.float64)
    outputs = NeuronStore("out", 6, mu=mu_out, initial_live=2, dtype=torch.float64)
    linear = CSTLinear(inputs, outputs, store, GaussianKernel(0.2).double())
    boundary = CSTBoundary(linear)
    root = QuotaRegime(
        budget=1,
        method=cSFW(backfit=None, pool_size=16, multistart=1),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
        quota=quota,
        interface=gamma_ungate(),
    )
    engine = StructuralEngine(
        {"layer": store, "in": inputs, "out": outputs},
        root,
        modules={"layer": linear},
        seed=5,
    )
    return engine, linear, boundary, outputs


def _one_event(engine, linear, boundary) -> tuple:
    x = torch.tensor(
        [[1.0, -0.5, 0.8, 0.2], [-0.3, 1.1, -0.7, 0.4]], dtype=torch.float64
    )
    upstream = torch.ones((2, 6), dtype=torch.float64)
    engine.begin_update()
    boundary(linear(x)).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    return engine.step()


def test_a_synthesized_quota_feeds_every_seat_that_can_spend_it() -> None:
    # A root handed a growing interface but no explicit quota used to
    # synthesize neuron_birth=0 and grow nothing, silently: a zero ceiling is
    # a legal quota, so there was no error to notice.
    engine, linear, boundary, outputs = _growing_world()

    operations = _one_event(engine, linear, boundary)

    assert any(isinstance(op, NeuronUngate) for op in operations)
    assert outputs.live_ids().numel() > 2


def test_an_explicit_quota_still_overrides_the_synthesized_one() -> None:
    engine, linear, boundary, outputs = _growing_world(
        quota=ConstantQuota(StructuralQuota(synapse_birth=1, neuron_birth=0))
    )

    operations = _one_event(engine, linear, boundary)

    assert not any(isinstance(op, NeuronUngate) for op in operations)
    assert outputs.live_ids().numel() == 2
