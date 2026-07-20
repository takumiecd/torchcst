"""GrowthByProfit: profit-gated forward construction (theory U-2's K* rule).

A RigL-style zero-amplitude birth has no effect on the objective until
polished (its factors carry no weight yet), so every test here supplies a
``polish`` callback: a few optimizer steps run *inside* the trial, after the
candidate atom applies and before the realized-profit read.  A rejected
trial rolls the polish back along with the birth (verified explicitly).
"""

from __future__ import annotations

from typing import Any

import torch

from torchcst.compute import RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import GrowthByProfit
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            _assert_equal(one, two)
    else:
        assert left == right


def _engine(price: float, *, capacity: int = 8, seed: int = 3):
    store = SynapseStore(
        "rank",
        2,
        2,
        capacity,
        spec=RepresentationSpec.rank_one(2, 2),
        dtype=torch.float64,
    )
    module = RankOneLinear(store, 2, 2)
    policy = GrowthByProfit(
        event_interval=1,
        atoms_per_event=1,
        price=price,
    )
    optimizer = torch.optim.Adam(store.parameters(), lr=0.05)
    engine = StructuralEngine(
        {"rank": store},
        policy,
        modules={"rank": module},
        optimizer=optimizer,
        seed=seed,
    )
    return store, module, optimizer, engine


def _objective_and_polish(
    module: RankOneLinear,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    *,
    polish_steps: int = 8,
):
    x = torch.eye(2, dtype=torch.float64)

    def objective() -> float:
        with torch.no_grad():
            residual = module(x) - x @ target.t()
            return float(residual.square().sum())

    def polish() -> None:
        for _ in range(polish_steps):
            optimizer.zero_grad()
            residual = module(x) - x @ target.t()
            loss = residual.square().sum()
            loss.backward()
            optimizer.step()

    return objective, polish


def test_zero_price_accepts_a_profitable_atom_and_reduces_the_dense_gap() -> None:
    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)
    store, module, optimizer, engine = _engine(price=0.0)
    objective, polish = _objective_and_polish(module, optimizer, target)
    before_gap = objective()

    ops = engine.step(objective, polish)

    assert any(isinstance(op, SynapseBirth) for op in ops)
    assert store.view().ids.numel() == 1
    assert objective() < before_gap
    assert len(engine.op_log()) == 1


def test_prohibitive_price_rejects_and_fully_restores_optimizer_and_store() -> None:
    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)
    store, module, optimizer, engine = _engine(price=1.0e6)
    objective, polish = _objective_and_polish(module, optimizer, target)
    store.view()  # materialize the lazy live_slots cache before snapshotting
    before_store = store.state_dict()
    before_optimizer = optimizer.state_dict()

    ops = engine.step(objective, polish)

    assert ops == ()
    assert store.view().ids.numel() == 0
    assert engine.op_log() == ()
    after_store = store.state_dict()
    # Every event ticks age, but AgeColumn.tick() only advances *live* slots
    # (see storage/mechanics.py); this store has zero live entities before
    # and after the rejected trial, so nothing to offset here, unlike
    # test_profit_court.py's single-live-atom scenario.
    _assert_equal(after_store, before_store)
    _assert_equal(optimizer.state_dict(), before_optimizer)


def test_growth_step_without_objective_raises() -> None:
    _, _, _, engine = _engine(price=0.0)

    try:
        engine.step()
    except RuntimeError as exc:
        assert "objective" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a missing objective")


def test_polish_without_objective_raises() -> None:
    _, module, optimizer, engine = _engine(price=0.0)
    _, polish = _objective_and_polish(
        module, optimizer, torch.eye(2, dtype=torch.float64)
    )

    try:
        engine.step(polish=polish)
    except RuntimeError as exc:
        assert "polish" in str(exc)
    else:
        raise AssertionError("expected RuntimeError: polish requires objective=")


def test_natural_stop_price_keeps_k_at_the_initial_seed_forever() -> None:
    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)
    _, module, optimizer, engine = _engine(price=1.0e6, seed=11)
    objective, polish = _objective_and_polish(module, optimizer, target)

    for _ in range(10):
        engine.step(objective, polish)

    assert engine.stores["rank"].view().ids.numel() == 0
    assert engine.op_log() == ()


def test_low_price_grows_k_monotonically_across_events() -> None:
    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)
    _, module, optimizer, engine = _engine(price=0.0, capacity=8, seed=5)
    objective, polish = _objective_and_polish(module, optimizer, target)

    k_trace = []
    for _ in range(6):
        engine.step(objective, polish)
        k_trace.append(engine.stores["rank"].view().ids.numel())

    assert k_trace == sorted(k_trace)
    assert k_trace[-1] > k_trace[0]


def test_higher_price_stops_growth_at_a_smaller_k_than_a_lower_price() -> None:
    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)

    def final_k(price: float, seed: int) -> int:
        _, module, optimizer, engine = _engine(price=price, capacity=8, seed=seed)
        objective, polish = _objective_and_polish(module, optimizer, target)
        for _ in range(40):
            engine.step(objective, polish)
        return engine.stores["rank"].view().ids.numel()

    low_price_k = final_k(price=1.0e-4, seed=7)
    high_price_k = final_k(price=0.05, seed=7)

    assert high_price_k < low_price_k
