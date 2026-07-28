"""Stage 3a: AbsorbPolicy whole-StructuralPolicy end-to-end wiring."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import AbsorbPolicy
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseAbsorb, SynapseBirth, SynapseStore


def _build(*, capacity: int = 8):
    """One CSTLinear with a planted near-duplicate triple plus one isolated atom.

    Atoms 0-2 sit within ``1e-4`` of ``(0.5, 0.5)`` -- far inside
    ``sigma=0.1`` and any of the small ``radius`` values these tests use, so
    they are near-identical rank-one directions. Atom 3 sits at ``(0.05,
    0.95)``, Euclidean ``z``-distance ``~0.636`` from the cluster -- outside
    every ``radius`` used below, so it is always screened out as a neighbor.
    """
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=capacity,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in",
        1,
        mu=torch.zeros(1, 1, dtype=torch.float64),
        initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.1).double())

    s = torch.tensor([[0.50], [0.5001], [0.4999], [0.05]], dtype=torch.float64)
    t = torch.tensor([[0.50], [0.5001], [0.4999], [0.95]], dtype=torch.float64)
    w = torch.tensor([1.0, -0.7, 0.4, 0.6], dtype=torch.float64)
    store.apply(
        [SynapseBirth(store.site, s, t, w, torch.arange(4, dtype=torch.int64))]
    )
    return store, inputs, outputs, module


def _train_step(engine: StructuralEngine, module: CSTLinear, optimizer) -> None:
    engine.begin_update()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    loss = ((module(x) - target) ** 2).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    engine.observe_microbatch(weight=1.0)
    engine.finalize_backward()
    optimizer.step()


def test_absorb_policy_e2e_absorbs_duplicates_preserves_function_and_trains_on() -> None:
    store, inputs, outputs, module = _build()
    optimizer = torch.optim.SGD(store.parameters(), lr=1.0e-5)
    policy = AbsorbPolicy(
        site="edge", event_interval=3, rent=1.0e-6, radius=0.01, ridge=1.0e-10
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        policy,
        modules={"edge": module},
        optimizer=optimizer,
    )

    # Two quiet updates (event_interval=3): no structural event yet.
    for _ in range(2):
        _train_step(engine, module, optimizer)
        applied = engine.step()
        assert applied == ()
    assert store.view().ids.numel() == 4

    # Third update lands on the cadence: the absorb event fires.
    _train_step(engine, module, optimizer)
    dense_before = module.dense_weight().detach().clone()
    applied = engine.step()

    absorbs = tuple(op for op in applied if isinstance(op, SynapseAbsorb))
    assert len(absorbs) == 2  # 3 near-duplicates -> 1 survivor is two absorbs
    assert store.view().ids.numel() == 2  # duplicate survivor + isolated atom

    dense_after = module.dense_weight().detach()
    torch.testing.assert_close(dense_before, dense_after, rtol=1.0e-4, atol=1.0e-7)

    assert len(policy.audit_log) == 2
    for entry in policy.audit_log:
        assert entry.cost <= policy.rent
        assert entry.receiver_count >= 1
        assert entry.res2_D >= 0.0
        assert entry.delta_w_frobenius >= 0.0
        assert entry.alpha_norm >= 0.0

    # Optimizer state stays consistent and training keeps working post-absorb.
    for _ in range(3):
        _train_step(engine, module, optimizer)
        engine.step()
    assert torch.isfinite(module.dense_weight()).all()


def test_absorb_policy_rent_gate_only_frees_near_zero_cost_absorbs() -> None:
    store, inputs, outputs, module = _build()
    isolated_id = int(store.view().ids[3])
    policy = AbsorbPolicy(
        site="edge",
        event_interval=1,
        rent=1.0e-9,
        radius=0.01,
        ridge=1.0e-10,
        budget=1000,  # generous budget: rent alone must gate acceptance
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        policy,
        modules={"edge": module},
    )
    engine.step()

    live_ids = store.view().ids.tolist()
    assert isolated_id in live_ids  # never absorbed despite the generous budget
    assert len(live_ids) == 2  # the duplicate trio still collapses for ~free
    for entry in policy.audit_log:
        assert entry.cost <= policy.rent


def test_absorb_policy_quiescent_event_emits_no_ops_and_does_not_error() -> None:
    store, inputs, outputs, module = _build()
    # Radius far below the planted duplicate spacing: nobody has a neighbor.
    policy = AbsorbPolicy(
        site="edge", event_interval=1, rent=1.0e9, radius=1.0e-12, ridge=1.0e-10
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        policy,
        modules={"edge": module},
    )
    before_ids = store.view().ids.clone()

    applied = engine.step()

    assert applied == ()
    assert engine.clock.event_index == 1  # still a (quiescent) event
    assert torch.equal(store.view().ids, before_ids)
    assert policy.audit_log == []


def test_absorb_policy_replay_is_deterministic() -> None:
    def run() -> tuple[tuple[int, object], ...]:
        store, inputs, outputs, module = _build()
        optimizer = torch.optim.SGD(store.parameters(), lr=1.0e-5)
        policy = AbsorbPolicy(
            site="edge", event_interval=1, rent=1.0e-6, radius=0.01, ridge=1.0e-10
        )
        engine = StructuralEngine(
            {"edge": store, "edge_in": inputs, "edge_out": outputs},
            policy,
            modules={"edge": module},
            optimizer=optimizer,
        )
        for _ in range(3):
            _train_step(engine, module, optimizer)
            engine.step()
        return engine.op_log()

    first = run()
    second = run()

    assert len(first) == len(second)
    for (index_a, op_a), (index_b, op_b) in zip(first, second):
        assert index_a == index_b
        assert type(op_a) is type(op_b)
        assert op_a.site == op_b.site
        assert op_a.dying == op_b.dying
        assert torch.equal(op_a.receivers, op_b.receivers)
        assert torch.equal(op_a.delta_w, op_b.delta_w)
