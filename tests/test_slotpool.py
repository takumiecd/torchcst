"""SlotPool の id/slot 帳簿付けの unit test。"""

from __future__ import annotations

import pytest
import torch

from torchcst.storage.common import (
    BalancedSlotPool,
    IdAllocator,
    SlotBirth,
    SlotDeath,
    SlotPool,
)


def test_id_allocator_never_reuse_monotonic():
    alloc = IdAllocator()
    a = alloc.issue(3)
    b = alloc.issue(2)
    assert a.dtype == torch.int64
    assert a.tolist() == [0, 1, 2]
    assert b.tolist() == [3, 4]
    # 単調増加 (重複なし)
    all_ids = torch.cat([a, b])
    assert all_ids.tolist() == sorted(all_ids.tolist())
    assert len(set(all_ids.tolist())) == all_ids.numel()


def test_id_allocator_rank_bits():
    alloc0 = IdAllocator(rank=0)
    alloc1 = IdAllocator(rank=1)
    id0 = alloc0.issue(1)
    id1 = alloc1.issue(1)
    # 異なる rank は上位ビットが異なるため衝突しない
    assert int(id0.item()) != int(id1.item())
    assert int(id1.item()) >> 48 == 1


def test_birth_death_then_birth_advances_id_space():
    pool = SlotPool(capacity=4)

    first = pool.apply([SlotBirth(2)])
    ids1, slots1 = first.born_ids, first.born_slots
    assert pool.k_live == 2

    pool.apply([SlotDeath(ids1)])
    assert pool.k_live == 0

    second = pool.apply([SlotBirth(2)])
    ids2, slots2 = second.born_ids, second.born_slots
    assert pool.k_live == 2
    # id 空間は進んでいる (再利用されない)
    assert set(ids1.tolist()).isdisjoint(set(ids2.tolist()))
    # slot は再利用されてよい (slot に意味はない: P1)
    assert set(slots2.tolist()) <= set(range(4))


def test_slots_of_and_ids_of_roundtrip():
    pool = SlotPool(capacity=8)
    change = pool.apply([SlotBirth(5)])
    ids, slots = change.born_ids, change.born_slots

    recovered_slots = pool.slots_of(ids)
    recovered_ids = pool.ids_of(recovered_slots)
    assert recovered_ids.tolist() == ids.tolist()

    # slot -> id -> slot も往復する
    back_slots = pool.slots_of(pool.ids_of(slots))
    assert set(back_slots.tolist()) == set(slots.tolist())


def test_slots_of_unknown_id_raises():
    pool = SlotPool(capacity=4)
    with pytest.raises(KeyError):
        pool.apply([SlotDeath(torch.tensor([999], dtype=torch.int64))])


def test_ids_of_dead_slot_raises():
    pool = SlotPool(capacity=4)
    change = pool.apply([SlotBirth(2)])
    slots = change.born_slots
    pool.apply([SlotDeath(change.born_ids)])
    with pytest.raises(KeyError):
        pool.ids_of(slots)


def test_capacity_exhausted_raises_runtime_error():
    pool = SlotPool(capacity=2)
    pool.apply([SlotBirth(2)])
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        pool.apply([SlotBirth(1)])


def test_live_slots_ascending_and_matches_k_live():
    pool = SlotPool(capacity=6)
    change = pool.apply([SlotBirth(4)])
    # 一部を death して穴を空ける
    pool.apply([SlotDeath(change.born_ids[1:2])])
    live = pool.live_slots
    assert live.tolist() == sorted(live.tolist())
    assert live.numel() == pool.k_live == 3


def test_compact_remap_is_consistent():
    pool = SlotPool(capacity=6)
    change = pool.apply([SlotBirth(4)])
    slots = change.born_slots
    # slot 0 と 2 を death して穴あきにする (残るのは slots[1], slots[3])
    dead = slots[[0, 2]]
    pool.apply([SlotDeath(pool.ids_of(dead))])
    live_ids_before = pool.ids_of(pool.live_slots)

    remap = pool.compact()
    assert remap.shape[0] == pool.capacity
    # 死んでいた slot は -1
    assert bool((remap[dead] == -1).all())

    # compact 後、live slot は 0..k_live-1 に前詰めされている
    live_after = pool.live_slots
    assert live_after.tolist() == list(range(pool.k_live))

    # id の対応関係 (どの id がどこにいるか) は remap を通して整合する
    ids_after = pool.ids_of(live_after)
    assert set(ids_after.tolist()) == set(live_ids_before.tolist())

    # 新たに割り当てれば残りの空き (k_live.. capacity-1) が使われる
    n_free = pool.capacity - pool.k_live
    more = pool.apply([SlotBirth(n_free)]).born_slots
    assert set(more.tolist()) == set(range(pool.capacity - n_free, pool.capacity))


def test_balanced_pool_bootstraps_then_requires_equal_birth_death_counts():
    pool = BalancedSlotPool(capacity=4)
    initial = pool.apply([SlotBirth(3)])

    change = pool.apply([
        SlotDeath(initial.born_ids[:2]),
        SlotBirth(2),
    ])
    assert change.dead_slots.numel() == change.born_slots.numel() == 2
    assert pool.k_live == 3

    ids_before = pool.ids_of(pool.live_slots).clone()
    with pytest.raises(ValueError, match="equal birth and death"):
        pool.apply([SlotBirth(1)])
    assert torch.equal(pool.ids_of(pool.live_slots), ids_before)


def test_balanced_pool_is_independent_and_keeps_only_packed_live_ids():
    pool = BalancedSlotPool(capacity=100)
    initial = pool.apply([SlotBirth(3)])

    assert not isinstance(pool, SlotPool)
    assert pool.live_slots.tolist() == [0, 1, 2]
    assert pool._ids_by_slot.numel() == 3
    assert not hasattr(pool, "_free_slots")
    assert not hasattr(pool, "_id_to_slot")
    assert not hasattr(pool, "_slot_to_id")

    dead_ids = initial.born_ids[[2, 0]]
    change = pool.apply([SlotDeath(dead_ids), SlotBirth(2)])
    assert change.born_slots.tolist() == [0, 2]
    assert pool.live_slots.tolist() == [0, 1, 2]


def test_failed_slot_batch_is_atomic_and_does_not_consume_ids():
    pool = SlotPool(capacity=3)
    initial = pool.apply([SlotBirth(1)])
    before_ids = pool.ids_of(pool.live_slots).clone()

    with pytest.raises(ValueError, match="duplicates"):
        pool.apply([
            SlotDeath(initial.born_ids.repeat(2)),
            SlotBirth(1),
        ])

    assert torch.equal(pool.ids_of(pool.live_slots), before_ids)
    assert pool.k_live == 1
    # 失敗した batch が ID を消費していなければ、次は 1 が発行される。
    change = pool.apply([SlotBirth(1)])
    assert change.born_ids.tolist() == [1]
