"""SlotPool の id/slot 帳簿付けの unit test。"""

from __future__ import annotations

import pytest
import torch

from torchcst.storage.common import IdAllocator, SlotPool


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


def test_allocate_release_then_reallocate_advances_id_space():
    alloc = IdAllocator()
    pool = SlotPool(capacity=4)

    ids1 = alloc.issue(2)
    slots1 = pool.allocate(ids1)
    assert pool.k_live == 2

    pool.release(slots1)
    assert pool.k_live == 0

    ids2 = alloc.issue(2)
    slots2 = pool.allocate(ids2)
    assert pool.k_live == 2
    # id 空間は進んでいる (再利用されない)
    assert set(ids1.tolist()).isdisjoint(set(ids2.tolist()))
    # slot は再利用されてよい (slot に意味はない: P1)
    assert set(slots2.tolist()) <= set(range(4))


def test_slots_of_and_ids_of_roundtrip():
    alloc = IdAllocator()
    pool = SlotPool(capacity=8)
    ids = alloc.issue(5)
    slots = pool.allocate(ids)

    recovered_slots = pool.slots_of(ids)
    recovered_ids = pool.ids_of(recovered_slots)
    assert recovered_ids.tolist() == ids.tolist()

    # slot -> id -> slot も往復する
    back_slots = pool.slots_of(pool.ids_of(slots))
    assert set(back_slots.tolist()) == set(slots.tolist())


def test_slots_of_unknown_id_raises():
    pool = SlotPool(capacity=4)
    with pytest.raises(KeyError):
        pool.slots_of(torch.tensor([999], dtype=torch.int64))


def test_ids_of_dead_slot_raises():
    alloc = IdAllocator()
    pool = SlotPool(capacity=4)
    ids = alloc.issue(2)
    slots = pool.allocate(ids)
    pool.release(slots)
    with pytest.raises(KeyError):
        pool.ids_of(slots)


def test_capacity_exhausted_raises_runtime_error():
    alloc = IdAllocator()
    pool = SlotPool(capacity=2)
    pool.allocate(alloc.issue(2))
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        pool.allocate(alloc.issue(1))


def test_live_slots_ascending_and_matches_k_live():
    alloc = IdAllocator()
    pool = SlotPool(capacity=6)
    ids = alloc.issue(4)
    slots = pool.allocate(ids)
    # 一部を release して穴を空ける
    pool.release(slots[1:2])
    live = pool.live_slots
    assert live.tolist() == sorted(live.tolist())
    assert live.numel() == pool.k_live == 3


def test_compact_remap_is_consistent():
    alloc = IdAllocator()
    pool = SlotPool(capacity=6)
    ids = alloc.issue(4)
    slots = pool.allocate(ids)
    # slot 0 と 2 を release して穴あきにする (残るのは slots[1], slots[3])
    dead = slots[[0, 2]]
    pool.release(dead)
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
    more = pool.allocate(alloc.issue(n_free))
    assert set(more.tolist()) == set(range(pool.k_live - n_free, pool.capacity))
