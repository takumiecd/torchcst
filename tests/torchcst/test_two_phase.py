"""v4 SynapseStore二相applyの原子性test。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from torchcst.storage import (
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
    commit_all,
    prepare_all,
)


def birth(site: str, n: int, lineage_start: int = 0) -> SynapseBirth:
    return SynapseBirth(
        site,
        s=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        t=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        w=torch.arange(1, n + 1, dtype=torch.float32),
        lineage=torch.arange(
            lineage_start, lineage_start + n, dtype=torch.int64
        ),
    )


@dataclass(frozen=True)
class Snapshot:
    version: int
    capacity: int
    ids: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor
    w: torch.Tensor
    age: torch.Tensor
    lineage: torch.Tensor
    next_id: int


def snapshot(store: SynapseStore) -> Snapshot:
    return Snapshot(
        version=store.version,
        capacity=store.capacity,
        ids=store.live_ids().clone(),
        s=store.s.detach().clone(),
        t=store.t.detach().clone(),
        w=store.w.detach().clone(),
        age=store.age.values.clone(),
        lineage=store.lineage.values.clone(),
        next_id=store._slots._allocator.next_id,
    )


def assert_unchanged(store: SynapseStore, before: Snapshot) -> None:
    after = snapshot(store)
    assert after.version == before.version
    assert after.capacity == before.capacity
    assert after.next_id == before.next_id
    for name in ("ids", "s", "t", "w", "age", "lineage"):
        assert torch.equal(getattr(after, name), getattr(before, name))


def invalid_batches(store: SynapseStore) -> list[tuple[list[object], type[Exception]]]:
    live = store.live_ids()
    wrong_shape = SynapseBirth(
        store.site,
        s=torch.zeros(1, 2, dtype=torch.int64),
        t=torch.zeros(1, 1, dtype=torch.int64),
        w=torch.zeros(1),
        lineage=torch.zeros(1, dtype=torch.int64),
    )
    return [
        (
            [
                SynapseDeath(store.site, live[:1]),
                SynapseDeath(store.site, torch.tensor([999], dtype=torch.int64)),
            ],
            KeyError,
        ),
        ([SynapseDeath(store.site, live[:1]), wrong_shape], ValueError),
        (
            [SynapseDeath(store.site, live[:1]), birth(store.site, 2, 20)],
            RuntimeError,
        ),
    ]


@pytest.mark.parametrize("case", [0, 1, 2])
def test_failed_prepare_leaves_every_store_bit_unchanged(case: int) -> None:
    store = SynapseStore("a", 1, 1, capacity=2, max_capacity=2)
    store.apply([birth("a", 2)])
    store.age.tick()
    before = snapshot(store)
    ops, error = invalid_batches(store)[case]

    with pytest.raises(error):
        store.prepare(ops)

    assert_unchanged(store, before)


def test_prepare_is_pure_and_ticket_snapshots_birth_values() -> None:
    store = SynapseStore("a", 1, 1, capacity=1)
    op = birth("a", 1, 5)
    before = snapshot(store)
    ticket = store.prepare([op])
    op.w.fill_(99.0)

    assert_unchanged(store, before)
    store.commit(ticket)
    assert store.view().w.item() == 1.0


def test_ticket_is_single_use_and_becomes_stale_after_another_commit() -> None:
    store = SynapseStore("a", 1, 1, capacity=2)
    first = store.prepare([birth("a", 1, 1)])
    stale = store.prepare([birth("a", 1, 2)])
    store.commit(first)

    with pytest.raises(RuntimeError, match="stale"):
        store.commit(stale)
    with pytest.raises(RuntimeError, match="already"):
        store.commit(first)


def test_cross_store_prepare_failure_commits_neither_store() -> None:
    left = SynapseStore("left", 1, 1, capacity=1, max_capacity=1)
    right = SynapseStore("right", 1, 1, capacity=1, max_capacity=1)
    left_before = snapshot(left)
    right_before = snapshot(right)

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        prepare_all(
            [
                (left, [birth("left", 1, 1)]),
                (right, [birth("right", 2, 2)]),
            ]
        )

    assert_unchanged(left, left_before)
    assert_unchanged(right, right_before)


def test_commit_all_applies_each_prevalidated_store_once() -> None:
    left = SynapseStore("left", 1, 1, capacity=1)
    right = SynapseStore("right", 1, 1, capacity=1)
    tickets = prepare_all(
        [
            (left, [birth("left", 1, 1)]),
            (right, [birth("right", 1, 2)]),
        ]
    )

    commit_all(tickets)

    assert left.version == right.version == 1
    assert left.view().ids.numel() == right.view().ids.numel() == 1
