"""Family-crossing LC policy and functional-mass acceptance tests."""

from __future__ import annotations

import torch

from torchcst.engine import StructuralEngine
from torchcst.policy.recipes import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def test_same_lc_policy_value_runs_entry_and_rank_one_stores() -> None:
    policy = LC(
        event_interval=1,
        birth_end_event=1,
        birth_budget=2,
        freeze_event=3,
    )
    entry = SynapseStore(
        "entry",
        1,
        1,
        1,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    rank_one = SynapseStore(
        "rank", 3, 2, 1, spec=RepresentationSpec.rank_one(3, 2)
    )

    entry_ops = StructuralEngine({"entry": entry}, policy, seed=21).step()
    rank_ops = StructuralEngine({"rank": rank_one}, policy, seed=21).step()

    assert any(isinstance(op, SynapseBirth) for op in entry_ops)
    assert any(isinstance(op, SynapseBirth) for op in rank_ops)
    assert entry.view().s.dtype == torch.int64
    assert rank_one.view().s.is_floating_point()
    torch.testing.assert_close(rank_one.view().s.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(rank_one.view().t.norm(dim=1), torch.ones(2))


def test_entry_and_gauged_rank_one_mass_both_reduce_to_absolute_weight() -> None:
    entry = SynapseStore(
        "entry", 1, 1, 1, spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1)
    )
    entry.apply(
        [
            SynapseBirth(
                "entry",
                torch.zeros(1, 1, dtype=torch.int64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.tensor([-2.5]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    rank_one = SynapseStore(
        "rank", 2, 2, 1, spec=RepresentationSpec.rank_one(2, 2)
    )
    rank_one.apply(
        [
            SynapseBirth(
                "rank",
                torch.tensor([[0.6, 0.8]]),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([-2.5]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )

    torch.testing.assert_close(entry.view().mass, entry.view().w.abs())
    torch.testing.assert_close(rank_one.view().mass, rank_one.view().w.abs())
