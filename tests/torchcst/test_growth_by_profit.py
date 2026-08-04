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

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy.cadences import PeriodicCadence
from torchcst.policy.contract import EvenBudgetDistributor
from torchcst.policy.families import SynapseLifecycle
from torchcst.policy.profit import ProfitCourt
from torchcst.policy.proposers import Bounds, UniformBirth
from torchcst.policy.tree import QuotaRegime
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


# The retired GrowthByProfit recipe, kept privately: this file's real
# subject is the engine's profit-trial contract (polish inside the trial,
# rollback of a rejected trial including its polish), which needs a
# profit-gated tree to exercise.
def _growth_by_profit(
    *,
    event_interval: int = 1,
    atoms_per_event: int = 1,
    price: float = 0.0,
    min_profit: float = 0.0,
    initial_weight: float = 0.0,
    bounds_in: Bounds | None = None,
    bounds_out: Bounds | None = None,
) -> QuotaRegime:
    """Greedy profit-gated forward construction (theory U-2) as a tree.

    Same parts as the retired ``catalog.GrowthByProfit`` preset: every event
    proposes ``atoms_per_event`` uniform candidates and the root's profit
    court keeps them only when the realized loss reduction beats the price.
    No prune -- the run's natural stop is U-2's predicted K*(price). The
    trial itself is the root's own subprotocol (docs/policy-tree-phase2.md
    ruling 2); the engine only lends its checkpoint mechanism.
    """

    def make_birth(lam: float | None) -> UniformBirth:
        del lam
        return UniformBirth(
            bounds_in=bounds_in,
            bounds_out=bounds_out,
            initial_weight=initial_weight,
        )

    method = SynapseLifecycle(
        birth_factory=make_birth,
        priceable=False,
        label="GrowthByProfit",
    )
    return QuotaRegime(
        budget=atoms_per_event,
        method=method,
        cadence=PeriodicCadence(event_interval=event_interval, freeze_event=None),
        distributor=EvenBudgetDistributor(),
        profit=ProfitCourt(min_profit=min_profit, cost_rate=price),
    )


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
        1,
        1,
        capacity,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "rank_in",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "rank_out",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    kernel = GaussianKernel(0.5).double()
    module = CSTLinear(inputs, outputs, store, kernel)
    policy = _growth_by_profit(
        event_interval=1,
        atoms_per_event=1,
        price=price,
    )
    optimizer = torch.optim.Adam(store.parameters(), lr=0.05)
    engine = StructuralEngine(
        {"rank": store, "rank_in": inputs, "rank_out": outputs},
        policy,
        modules={"rank": module},
        optimizer=optimizer,
        seed=seed,
    )
    return store, module, optimizer, engine


def _objective_and_polish(
    module: CSTLinear,
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
