"""Canonical rank-one merge semantics and atomic store application."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from cstf.compute import RankOneLinear
from cstf.policy import MergeProposer, RetiredCandidateRegistry
from cstf.representation import RepresentationSpec
from cstf.storage import SynapseBirth, SynapseMerge, SynapseStore


def _rank_store(
    source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> SynapseStore:
    store = SynapseStore(
        "rank",
        source.shape[1],
        target.shape[1],
        source.shape[0],
        spec=RepresentationSpec.rank_one(source.shape[1], target.shape[1]),
        dtype=source.dtype,
    )
    store.apply(
        [
            SynapseBirth(
                "rank",
                source,
                target,
                weights,
                torch.arange(10, 10 + source.shape[0], dtype=torch.int64),
            )
        ]
    )
    return store


def test_merge_dense_residual_is_the_analytic_best_rank_one_residual() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0], [0.8, -0.6]], dtype=torch.float64)
    weights = torch.tensor([1.25, -0.7], dtype=torch.float64)
    store = _rank_store(source, target, weights)
    module = RankOneLinear(store, 3, 2)
    dense_before = module.dense_weight().detach().clone()
    singular = torch.linalg.svdvals(dense_before)
    parent_lineages = set(store.view().lineages.tolist())

    store.apply([SynapseMerge("rank", store.view().ids.reshape(1, 2))])

    dense_after = module.dense_weight().detach()
    torch.testing.assert_close(
        torch.linalg.matrix_norm(dense_before - dense_after), singular[1]
    )
    assert store.view().ids.numel() == 1
    assert int(store.view().lineages[0]) not in parent_lineages


def test_entry_merge_is_explicitly_undefined() -> None:
    store = SynapseStore("entry", 1, 1, 2)
    with pytest.raises(NotImplementedError, match="discrete lattice"):
        store.prepare(
            [SynapseMerge("entry", torch.zeros((0, 2), dtype=torch.int64))]
        )


@dataclass(frozen=True)
class _Snapshot:
    state: dict[str, object]
    version: int
    ids: torch.Tensor


def test_merge_prepare_failure_is_bit_atomic() -> None:
    store = _rank_store(
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    before = _Snapshot(store.state_dict(), store.version, store.live_ids().clone())
    pairs = torch.tensor(
        [[int(store.live_ids()[0]), int(store.live_ids()[1])], [999, 1000]],
        dtype=torch.int64,
    )

    with pytest.raises(KeyError):
        store.prepare([SynapseMerge("rank", pairs)])

    assert store.version == before.version
    assert torch.equal(store.live_ids(), before.ids)
    for name in ("s", "t", "w"):
        assert torch.equal(store.state_dict()[name], before.state[name])


def test_merge_proposer_returns_top_disjoint_pairs_with_pair_budget() -> None:
    source = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    target = source.clone()
    store = _rank_store(source, target, torch.ones(4, dtype=torch.float64))
    proposer = MergeProposer(similarity_threshold=0.99)

    proposed = proposer.propose(
        store.view(),
        2,
        RetiredCandidateRegistry(),
        torch.Generator().manual_seed(1),
    )

    assert len(proposed) == 1
    pairs = proposed[0].id_pairs
    assert pairs.shape == (2, 2)
    assert pairs.reshape(-1).unique().numel() == 4

