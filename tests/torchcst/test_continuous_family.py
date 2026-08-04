"""Continuous family integration with policy, capture, bounds, and replay."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.lab import Ledger, RngStreams
from torchcst.policy import (
    EvenBudgetDistributor,
    InstrumentSpec,
    MagnitudeCourt,
    PeriodicCadence,
    QuotaRegime,
    RetiredCandidateRegistry,
    SynapseLifecycle,
)
from tests.torchcst._recipes import LC
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


def _continuous_store(site: str = "continuous", capacity: int = 2) -> SynapseStore:
    return SynapseStore(
        site,
        1,
        1,
        capacity,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )


def test_same_lc_policy_value_completes_all_three_families() -> None:
    policy = LC(
        event_interval=1,
        birth_end_event=1,
        birth_budget=2,
        freeze_event=3,
        initial_weight=1.0,
    )
    entry = SynapseStore(
        "entry", 1, 1, 1, spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2)
    )
    rank = SynapseStore("rank", 2, 2, 1, spec=RepresentationSpec.rank_one(2, 2))
    continuous = _continuous_store(capacity=1)

    results = tuple(
        StructuralEngine({store.site: store}, policy, seed=17).step()
        for store in (entry, rank, continuous)
    )

    assert all(any(isinstance(op, SynapseBirth) for op in ops) for ops in results)
    assert entry.view().s.dtype == torch.int64
    torch.testing.assert_close(rank.view().s.norm(dim=1), torch.ones(2))
    assert bool(((continuous.view().s >= 0.0) & (continuous.view().s <= 1.0)).all())


def test_continuous_lifecycle_honors_immunity_rent_and_quiescence() -> None:
    store = _continuous_store(capacity=1)
    engine = StructuralEngine(
        {store.site: store},
        LC(
            event_interval=1,
            birth_end_event=4,
            birth_budget=2,
            freeze_event=8,
            initial_weight=1.0,
        ),
        seed=5,
    )
    first = engine.step()
    weak_lineage = next(op for op in first if isinstance(op, SynapseBirth)).lineage[0]

    events = [first]
    for _ in range(7):
        view = store.view()
        slots = store._slots.slots_of(view.ids)
        with torch.no_grad():
            store.w.index_fill_(0, slots.to(store.w.device), 10.0)
            weak = slots[store.lineage.values.index_select(0, slots) == weak_lineage]
            if weak.numel():
                store.w.index_fill_(0, weak.to(store.w.device), 0.01)
        events.append(engine.step())

    deaths = [op for event in events for op in event if isinstance(op, SynapseDeath)]
    assert all(
        not any(isinstance(op, SynapseDeath) for op in event) for event in events[:4]
    )
    assert any(int(weak_lineage) in death.ids.tolist() for death in deaths)
    assert all(not event for event in events[5:])


def test_engine_clamps_box_coordinates_after_optimizer_step() -> None:
    store = _continuous_store()
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.1], [0.9]], dtype=torch.float64),
                torch.tensor([[0.2], [0.8]], dtype=torch.float64),
                torch.ones(2, dtype=torch.float64),
                torch.arange(2, dtype=torch.int64),
            )
        ]
    )
    optimizer = torch.optim.SGD(store.parameters(), lr=4.0, momentum=0.9)
    engine = StructuralEngine(
        {store.site: store},
        LC(event_interval=100, birth_budget=0),
        optimizer=optimizer,
    )

    optimizer.zero_grad()
    (-(store.s.sum() + store.t.sum())).backward()
    optimizer.step()
    engine.step()

    view = store.view()
    assert bool(((view.s >= 0.0) & (view.s <= 1.0)).all())
    assert bool(((view.t >= 0.0) & (view.t <= 1.0)).all())


@dataclass
class _GradFieldRequest:
    requires: tuple[InstrumentSpec, ...] = field(
        default=(InstrumentSpec("grad_field_ema", decay=0.0),), init=False
    )

    def propose(self, view, budget, registry: RetiredCandidateRegistry, rng):
        return ()


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_cst_capture_gradfield_matches_autograd_weight_gradient(
    capture_mode: str,
) -> None:
    store = _continuous_store()
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.2], [0.75]], dtype=torch.float64),
                torch.tensor([[0.35], [0.9]], dtype=torch.float64),
                torch.tensor([0.7, -0.4], dtype=torch.float64),
                torch.arange(2, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        3,
        mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64),
        initial_live=3,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        2,
        mu=torch.tensor([[0.1], [0.8]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.3).double())
    method = SynapseLifecycle(
        birth_factory=lambda lam: _GradFieldRequest(),
        prune_factory=lambda: MagnitudeCourt(0.0),
        priceable=False,
        label="grad-field-observer",
    )
    policy = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        policy,
        modules={store.site: module},
        capture_mode=capture_mode,
    )
    x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.25, 0.75]], dtype=torch.float64)
    upstream = torch.tensor([[0.5, -2.0], [1.5, 0.25]], dtype=torch.float64)

    engine.begin_update()
    module(x).backward(upstream)
    expected = store.w.grad.index_select(0, store._slots.live_slots).abs().clone()
    engine.observe_microbatch()
    engine.finalize_backward()
    ids, scores = engine.instrument(store.site, "grad_field_ema").snapshot()

    assert torch.equal(ids, store.view().ids)
    torch.testing.assert_close(scores, expected)


def _continuous_replay(root_seed: int) -> str:
    store = _continuous_store(capacity=1)
    engine = StructuralEngine(
        {store.site: store},
        LC(
            event_interval=1,
            birth_end_event=3,
            birth_budget=2,
            freeze_event=5,
            initial_weight=1.0,
        ),
        rng=RngStreams(root_seed).get("proposal"),
    )
    for _ in range(5):
        engine.step()
    canonical = [
        {"event_index": event, "op": Ledger.serialize_ops((op,))[0]}
        for event, op in engine.op_log()
    ]
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def test_continuous_two_phase_structure_replay_is_bit_identical() -> None:
    assert _continuous_replay(2026) == _continuous_replay(2026)
    assert _continuous_replay(2026) != _continuous_replay(2027)
