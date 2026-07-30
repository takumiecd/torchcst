"""Deterministic reduced 5c H2 response lifecycle."""

from __future__ import annotations

import torch

from torchcst.compute import EntryLinear, NeuronGatedLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import Clock, Phase, UniformBirth
from torchcst.policy.recipes import LC_response
from torchcst.representation import RepresentationSpec
from torchcst.storage import (
    RETIRED,
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def test_scripted_response_bundle_immunity_rent_cascade_and_requiescence() -> None:
    synapses = SynapseStore(
        "edge",
        1,
        1,
        6,
        spec=RepresentationSpec.entry(bounds_in=3, bounds_out=2),
    )
    synapses.apply(
        [
            SynapseBirth(
                "edge",
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([10.0]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    outputs = NeuronStore("outputs", 2, initial_live=1)
    module = NeuronGatedLinear(EntryLinear(synapses, 3, 2), out_neurons=outputs)
    policy = LC_response(
        event_interval=1,
        birth_end_event=1,
        birth_budget=0,
        freeze_event=2,
        response_events=(3, 3),
        response_ungates_per_event=1,
        response_birth_budget=2,
        incident_births=2,
        immunity_events=2,
        strikes=2,
        bounds_in=3,
        bounds_out=2,
        initial_weight=1.0,
        initial_gate=1.0e-3,
    )
    engine = StructuralEngine(
        {"edge": synapses, "outputs": outputs},
        policy,
        modules={"edge": module},
        seed=7,
    )

    assert engine.step() == ()  # bounded initial window has zero supply
    assert engine.step() == ()  # frozen stationary phase before known switch
    response = engine.step()
    assert [type(op) for op in response] == [NeuronUngate, SynapseBirth]
    assert response[1].w.numel() == 2
    assert response[1].t[:, 0].tolist() == [1, 1]

    # Keep the neuron useful while one incident atom demonstrates rent cleanup.
    with torch.no_grad():
        outputs.gate[1] = 1.0
        incident_slots = synapses._slots.live_slots[
            synapses.t.index_select(0, synapses._slots.live_slots)[:, 0] == 1
        ]
        synapses.w[incident_slots[0]] = 0.1

    immune = engine.step()
    first_synapse_strike = engine.step()
    weak_prune = engine.step()
    assert immune == ()  # age < immunity_events is a runtime no-prune zone
    assert first_synapse_strike == ()
    assert len([op for op in weak_prune if isinstance(op, SynapseDeath)]) == 1
    assert outputs.state[1].item() != RETIRED

    # Once the gate becomes weak, two neuron strikes retire it.  The remaining
    # incident atom dies in the exact same cross-store plan by endpoint cascade.
    with torch.no_grad():
        outputs.gate[1] = 1.0e-3
    assert engine.step() == ()
    retired = engine.step()
    # Op order within one applied event changed under the tree-native
    # runtime: EventPlan.flattened() (policy/tree.py) is phase-major
    # (absorbs, deaths, retires, ungates, births), so the cascaded
    # SynapseDeath now precedes the NeuronRetire that triggered it, the
    # reverse of the old composed-engine's op-log order. This is a pure
    # op_log bit-compatibility change, not a behavior change (op_log bit
    # compatibility was explicitly retired in favor of statistical
    # equivalence -- docs/policy-tree-phase2.md, pinned by
    # tools/phase2_baseline.json); both ops are still in the same applied
    # set for the same event.
    assert [type(op) for op in retired] == [SynapseDeath, NeuronRetire]
    assert outputs.state[1].item() == RETIRED
    assert not bool((synapses.view().t[:, 0] == 1).any())

    assert engine.step() == ()
    assert engine.step() == ()
    assert all(
        not isinstance(op, SynapseBirth)
        for event_index, op in engine.op_log()
        if event_index > 3
    )

    # Even a free-standing coverage proposer sees the retired output row as
    # outside its candidate universe. There is exactly one synapse child
    # bound in this tree ("edge"); its view() is the same construction a
    # birth proposer would receive from the tree's own propose() path.
    proposal_view = engine._tree.children[0].view()
    proposed = UniformBirth().propose(proposal_view, 6, engine.registry, engine.rng)
    assert not proposed or not bool((proposed[0].t[:, 0] == 1).any())


def test_response_cadence_and_quota_separate_timing_from_supply() -> None:
    policy = LC_response(
        event_interval=1,
        birth_end_event=1,
        freeze_event=2,
        response_events=(4, 5),
        response_ungates_per_event=2,
        response_birth_budget=7,
        incident_births=3,
    )
    clock = Clock(update_step=4, event_index=4)
    signal = policy.cadence.event(clock)

    assert signal is not None
    assert signal.phase is Phase.RESPONSE
    quota = policy.quota.at(clock, signal.phase)
    assert quota.neuron_birth == 2
    assert quota.synapse_birth == 7
