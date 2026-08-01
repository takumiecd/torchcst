"""Damped acceptance: re-offer a rejected insertion at a smaller magnitude.

A solved amplitude is optimal under a local model that knows nothing about the
rest of training, so late in training entering at full strength is what FC-1
found to be a bomb. The plain trial is all-or-nothing, which throws away an
insertion that would have paid at half its size. Each ladder rung re-offers the
same operations scaled down; the first that actually pays is kept.
"""

from __future__ import annotations

import torch

from torchcst.compute import EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    EvenBudgetDistributor,
    PeriodicCadence,
    ProfitCourt,
    QuotaRegime,
    SynapseLifecycle,
)
from torchcst.representation import RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

TARGET = 0.5


class _FixedBirth:
    """Always propose one entry atom carrying weight 1.0 -- twice the optimum."""

    requires: tuple = ()

    def propose(self, view, budget, registry, rng):
        del registry, rng
        if budget == 0 or view.ids.numel():
            return ()
        return (
            SynapseBirth(
                view.site,
                torch.zeros((1, 1), dtype=torch.int64),
                torch.zeros((1, 1), dtype=torch.int64),
                torch.ones(1, dtype=torch.float64),
                torch.tensor([0], dtype=torch.int64),
            ),
        )


def _engine(damping: tuple[float, ...], *, min_profit: float = 0.0):
    store = SynapseStore(
        "edge", 1, 1, 4,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "in", 1, mu=torch.zeros(1, 1, dtype=torch.float64),
        initial_live=1, dtype=torch.float64,
    )
    outputs = NeuronStore(
        "out", 1, mu=torch.zeros(1, 1, dtype=torch.float64),
        initial_live=1, dtype=torch.float64,
    )
    module = EntryLinear(store, 1, 1)
    root = QuotaRegime(
        budget=1,
        method=SynapseLifecycle(
            birth_factory=lambda lam: _FixedBirth(),
            priceable=False,
            label="fixed",
        ),
        cadence=PeriodicCadence(event_interval=1),
        distributor=EvenBudgetDistributor(),
        profit=ProfitCourt(min_profit=min_profit, damping=damping),
    )
    engine = StructuralEngine(
        {"edge": store, "in": inputs, "out": outputs},
        root,
        modules={"edge": module},
        seed=1,
    )

    def objective() -> float:
        # Minimized at a total amplitude of exactly TARGET, so the proposed
        # 1.0 lands exactly as far away as proposing nothing: zero profit,
        # rejected. Half of it is perfect.
        with torch.no_grad():
            return abs(float(store.w.sum()) - TARGET)

    return engine, store, objective


def test_full_strength_insertion_is_rejected_without_a_ladder() -> None:
    engine, store, objective = _engine(())

    assert engine.step(objective) == ()
    assert store.live_ids().numel() == 0


def test_a_rejected_insertion_is_re_offered_at_a_smaller_magnitude() -> None:
    engine, store, objective = _engine((1.0, 0.5, 0.25))

    operations = engine.step(objective)

    births = [op for op in operations if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    # The rung that paid is the one that lands on the optimum, and it is the
    # damped values -- not the originally proposed ones -- that get committed.
    torch.testing.assert_close(
        births[0].w, torch.tensor([0.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        store.w.sum(), torch.tensor(0.5, dtype=torch.float64)
    )


def test_a_ladder_that_never_pays_still_commits_nothing() -> None:
    # The best any rung can do here is a profit of TARGET; demand more and
    # every rung rolls back, leaving the store exactly as it was.
    engine, store, objective = _engine((1.0, 0.5, 0.25), min_profit=0.6)

    operations = engine.step(objective)

    assert operations == ()
    assert store.live_ids().numel() == 0
    assert float(store.w.sum()) == 0.0


def test_damping_rungs_must_strictly_decrease() -> None:
    for bad in ((0.5, 0.5), (0.5, 0.9), (1.5,), (0.0,)):
        try:
            ProfitCourt(damping=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for damping={bad!r}")
