"""SynapseStore の apply (birth/death) と follower 通知の test。"""

from __future__ import annotations

import pytest
import torch

from torchcst.storage.base import Follower
from torchcst.storage.common import BalancedSlotPool
from torchcst.storage.synapse import (
    SynapseBirth,
    SynapseDeath,
    SynapseKick,
    SynapseMerge,
    SynapseStore,
)


class RecordingFollower(Follower):
    """記録用ダミー Follower。"""

    def __init__(self):
        self.grown: list[int] = []
        self.births: list[list[int]] = []
        self.deaths: list[list[int]] = []
        self.merges: list[tuple] = []
        self.remaps: list[list[int]] = []

    def grow(self, new_capacity: int) -> None:
        self.grown.append(new_capacity)

    def on_birth(self, slots: torch.Tensor) -> None:
        self.births.append(slots.tolist())

    def on_death(self, slots: torch.Tensor) -> None:
        self.deaths.append(slots.tolist())

    def on_merge(self, src_slots, dst_slots, mass) -> None:
        self.merges.append((src_slots.tolist(), dst_slots.tolist()))

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        self.remaps.append(old_to_new.tolist())


def make_store(capacity=16, d_in=1, d_out=1):
    return SynapseStore("l1", d_in=d_in, d_out=d_out, capacity=capacity)


def test_birth_increases_k_live_and_version():
    store = make_store()
    assert store.version == 0
    K = 5
    op = SynapseBirth("l1", s=torch.randn(K, 1), t=torch.randn(K, 1), w=torch.zeros(K))
    store.apply([op])
    assert store.version == 1
    view = store.view()
    assert view.ids.numel() == K


def test_death_decreases_k_live_and_version():
    store = make_store()
    K = 5
    store.apply([SynapseBirth(
        "l1", s=torch.randn(K, 1), t=torch.randn(K, 1), w=torch.zeros(K)
    )])
    ids = store.view().ids
    store.apply([SynapseDeath("l1", ids=ids[:2])])
    assert store.version == 2
    view = store.view()
    assert view.ids.numel() == K - 2


def test_death_removes_id_from_view():
    store = make_store()
    K = 4
    store.apply([SynapseBirth(
        "l1", s=torch.randn(K, 1), t=torch.randn(K, 1), w=torch.zeros(K)
    )])
    ids = store.view().ids
    dead_id = ids[1:2]
    store.apply([SynapseDeath("l1", ids=dead_id)])
    remaining = store.view().ids
    assert int(dead_id.item()) not in remaining.tolist()


def test_follower_receives_birth_and_death_notifications():
    store = make_store()
    follower = RecordingFollower()
    store.followers().subscribe(follower)

    K = 3
    store.apply([SynapseBirth(
        "l1", s=torch.randn(K, 1), t=torch.randn(K, 1), w=torch.zeros(K)
    )])
    assert len(follower.births) == 1
    assert len(follower.births[0]) == K

    ids = store.view().ids
    store.apply([SynapseDeath("l1", ids=ids[:1])])
    assert len(follower.deaths) == 1
    assert len(follower.deaths[0]) == 1


def test_death_birth_batch_preserves_k_live_and_increments_version_once():
    store = make_store(capacity=3)
    store.apply([SynapseBirth(
        "l1", s=torch.randn(3, 1), t=torch.randn(3, 1), w=torch.ones(3)
    )])
    old_ids = store.view().ids.clone()

    store.apply([
        SynapseDeath("l1", ids=old_ids[:1]),
        SynapseBirth(
            "l1", s=torch.randn(1, 1), t=torch.randn(1, 1), w=torch.zeros(1)
        ),
    ])

    assert store.version == 2
    assert store._slots.k_live == 3
    assert int(old_ids[0].item()) not in store.view().ids.tolist()


def test_invalid_batch_is_rejected_before_any_op_is_applied():
    store = make_store(capacity=2)
    store.apply([SynapseBirth(
        "l1", s=torch.randn(2, 1), t=torch.randn(2, 1), w=torch.ones(2)
    )])
    before = store.canonical_state()

    with pytest.raises(ValueError, match="last dim"):
        store.apply([
            SynapseDeath("l1", ids=before["ids"][:1]),
            SynapseBirth(
                "l1", s=torch.randn(1, 2), t=torch.randn(1, 1),
                w=torch.zeros(1),
            ),
        ])

    after = store.canonical_state()
    assert store.version == 1
    for key in before:
        assert torch.equal(before[key], after[key])


def test_balanced_slot_pool_validates_store_birth_death_batch():
    store = SynapseStore(
        "l1", d_in=1, d_out=1, capacity=3,
        slot_pool=BalancedSlotPool(3),
    )
    store.apply([SynapseBirth(
        "l1", s=torch.randn(3, 1), t=torch.randn(3, 1), w=torch.ones(3)
    )])
    ids = store.view().ids.clone()

    store.apply([
        SynapseDeath("l1", ids=ids[:1]),
        SynapseBirth(
            "l1", s=torch.randn(1, 1), t=torch.randn(1, 1), w=torch.zeros(1)
        ),
    ])
    assert store.view().ids.numel() == 3

    with pytest.raises(ValueError, match="equal birth and death"):
        store.apply([SynapseDeath("l1", ids=store.view().ids[:1])])
    assert store.view().ids.numel() == 3


def test_apply_unsupported_op_raises_type_error():
    store = make_store()

    class Bogus:
        site = "l1"

    with pytest.raises(TypeError):
        store.apply([Bogus()])


def test_merge_and_kick_raise_not_implemented():
    store = make_store()
    with pytest.raises(NotImplementedError):
        store.apply([SynapseMerge(
            "l1", id_pairs=torch.zeros(0, 2, dtype=torch.int64)
        )])
    with pytest.raises(NotImplementedError):
        store.apply([SynapseKick("l1", ids=torch.zeros(0, dtype=torch.int64))])


def test_capacity_exhausted_raises_runtime_error():
    store = make_store(capacity=3)
    store.apply([SynapseBirth(
        "l1", s=torch.randn(3, 1), t=torch.randn(3, 1), w=torch.zeros(3)
    )])
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        store.apply([SynapseBirth(
            "l1", s=torch.randn(1, 1), t=torch.randn(1, 1), w=torch.zeros(1)
        )])
