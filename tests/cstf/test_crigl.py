"""cRigL control-arm integration and accumulation invariance."""

from __future__ import annotations

import torch

from cstf.compute import EntryLinear
from cstf.engine import StructuralEngine
from cstf.lab import Ledger
from cstf.policy import LC, cRigL
from cstf.representation import RepresentationSpec
from cstf.storage import SynapseBirth, SynapseStore


def _engine(policy):
    store = SynapseStore(
        "entry",
        1,
        1,
        3,
        spec=RepresentationSpec.entry(bounds_in=3, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([0.1, 2.0]),
                torch.tensor([0, 3], dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 3, 2)
    return store, module, StructuralEngine(
        {"entry": store}, policy, modules={"entry": module}, seed=19
    )


def _run(chunks: int):
    policy = cRigL(
        event_interval=1,
        observe_window=1,
        drop_fraction=0.5,
        pool_size=99,
        decay=0.0,
    )
    _, module, engine = _engine(policy)
    x = torch.tensor(
        [[1.0, 2.0, 4.0], [3.0, -1.0, 2.0], [2.0, 1.0, -2.0], [0.0, 3.0, 1.0]]
    )
    g = torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, -2.0], [1.0, 3.0]])
    engine.begin_update()
    if chunks == 1:
        module(x).backward(g)
        engine.observe_microbatch(weight=1.0)
    else:
        for x_part, g_part in zip(x.chunk(2), g.chunk(2)):
            module(x_part).backward(2.0 * g_part)
            engine.observe_microbatch(weight=0.5)
    engine.finalize_backward()
    snapshot = engine.instruments["entry"]["candidate_field"].snapshot()
    ops = engine.step()
    serialized = Ledger.serialize_ops(ops)
    return snapshot, serialized


def test_abs_after_sum_is_gradient_accumulation_invariant() -> None:
    (coords_one, scores_one), log_one = _run(1)
    (coords_two, scores_two), log_two = _run(2)
    assert torch.equal(coords_one, coords_two)
    torch.testing.assert_close(scores_one, scores_two)
    assert log_one == log_two


def test_scripted_gradient_selects_the_unique_large_coordinate() -> None:
    policy = cRigL(
        event_interval=1,
        observe_window=1,
        drop_fraction=0.5,
        pool_size=99,
        decay=0.0,
    )
    _, module, engine = _engine(policy)
    engine.begin_update()
    x = torch.tensor([[0.0, 0.0, 10.0]], requires_grad=True)
    module(x).backward(torch.tensor([[0.0, 10.0]]))
    engine.observe_microbatch()
    engine.finalize_backward()
    ops = engine.step()
    birth = next(op for op in ops if isinstance(op, SynapseBirth))
    assert birth.s.tolist() == [[2]]
    assert birth.t.tolist() == [[1]]
    assert birth.w.item() == 0.0


def test_policy_swap_uses_one_engine_update_api_with_and_without_capture() -> None:
    for policy, expected_capture in (
        (LC(event_interval=1, birth_budget=0), False),
        (cRigL(event_interval=1, observe_window=1, pool_size=4), True),
    ):
        _, module, engine = _engine(policy)
        engine.begin_update()
        assert engine.capture_active is expected_capture
        module(torch.randn(2, 3, requires_grad=True)).sum().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        assert isinstance(engine.step(), tuple)
