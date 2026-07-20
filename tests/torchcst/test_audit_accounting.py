"""Representation-price and measured optimizer-state accounting."""

from __future__ import annotations

import pytest
import torch

from torchcst.audit import Accounting
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def _entry_store() -> SynapseStore:
    store = SynapseStore(
        "entry",
        1,
        1,
        2,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.ones(2),
                torch.tensor([0, 3], dtype=torch.int64),
            )
        ]
    )
    return store


def _rank_one_store(m: int, n: int) -> SynapseStore:
    store = SynapseStore(
        "rank", n, m, 2, spec=RepresentationSpec.rank_one(n, m)
    )
    source = torch.randn(2, n)
    target = torch.randn(2, m)
    source /= torch.linalg.vector_norm(source, dim=1, keepdim=True)
    target /= torch.linalg.vector_norm(target, dim=1, keepdim=True)
    store.apply(
        [
            SynapseBirth(
                "rank",
                source,
                target,
                torch.ones(2),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    return store


def test_active_params_use_representation_atom_price() -> None:
    m, n = 5, 3
    entry = _entry_store()
    rank = _rank_one_store(m, n)
    report = Accounting.active_params({"entry": entry, "rank": rank})

    assert report.by_site == {"entry": 2, "rank": 2 * (m + n + 1)}
    assert report.total == 2 + 2 * (m + n + 1)


def test_adam_state_bytes_are_measured_from_materialized_state() -> None:
    store = _rank_one_store(4, 3)
    optimizer = torch.optim.Adam(store.parameters(), lr=0.01)
    loss = store.s.sum() + store.t.sum() + store.w.sum()
    loss.backward()
    optimizer.step()

    expected = sum(
        int(value.nbytes)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert expected > 0
    assert Accounting.optimizer_state_bytes(optimizer) == expected


def test_rank_one_gamma_reproduces_granularity_lemma() -> None:
    m, n = 7, 11
    spec = RepresentationSpec.rank_one(n, m)

    assert Accounting.gamma(spec, m, n) == pytest.approx((m + n + 1) / (m * n))
    assert Accounting.gamma(RepresentationSpec.entry(), m, n) == pytest.approx(
        1 / (m * n)
    )
