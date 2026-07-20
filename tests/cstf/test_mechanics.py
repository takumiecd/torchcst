"""v4 storage mechanics契約の決定的test。"""

from __future__ import annotations

import torch

from cstf.storage import (
    IdAllocator,
    SlotBirth,
    SlotDeath,
    SlotPool,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


class DummyMomentColumn:
    """optimizer moment相当のslot-indexedダミーfollower。"""

    def __init__(self) -> None:
        self.values = torch.zeros(0)

    def grow(self, new_capacity: int) -> None:
        grown = torch.zeros(new_capacity)
        grown[: self.values.numel()] = self.values
        self.values = grown

    def on_birth(self, slots: torch.Tensor, lineage: torch.Tensor) -> None:
        self.values[slots] = 0.0

    def on_death(self, slots: torch.Tensor) -> None:
        self.values[slots] = 0.0

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        remapped = torch.zeros_like(self.values)
        old = torch.nonzero(old_to_new >= 0, as_tuple=False).flatten()
        remapped[old_to_new[old]] = self.values[old]
        self.values = remapped


def birth(site: str, lineage: list[int]) -> SynapseBirth:
    n = len(lineage)
    return SynapseBirth(
        site,
        s=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        t=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        w=torch.ones(n),
        lineage=torch.tensor(lineage, dtype=torch.int64),
    )


def test_id_allocator_is_int64_monotonic_and_never_reuses() -> None:
    allocator = IdAllocator()
    first = allocator.issue(3)
    preview = allocator.preview(2)
    second = allocator.issue(2)

    assert first.dtype == second.dtype == torch.int64
    assert first.tolist() == [0, 1, 2]
    assert preview.tolist() == second.tolist() == [3, 4]
    assert allocator.next_id == 5


def test_slot_pool_reuses_slots_but_never_ids_and_doubles_capacity() -> None:
    pool = SlotPool(2)
    initial = pool.apply([SlotBirth(2)])
    dead_slot = initial.born_slots[:1].clone()
    replacement = pool.apply(
        [SlotDeath(initial.born_ids[:1]), SlotBirth(1)]
    )

    assert replacement.born_slots.tolist() == dead_slot.tolist()
    assert set(initial.born_ids.tolist()).isdisjoint(replacement.born_ids.tolist())
    grown = pool.apply([SlotBirth(2)])
    assert pool.capacity == 4
    assert grown.new_capacity == 4
    assert torch.equal(pool.ids_of(pool.slots_of(grown.born_ids)), grown.born_ids)


def test_live_slots_cache_is_rebuilt_only_after_version_change() -> None:
    pool = SlotPool(2)
    first = pool.live_slots
    second = pool.live_slots
    pool.prepare([SlotBirth(1)])
    third = pool.live_slots

    assert first is second is third
    pool.apply([SlotBirth(1)])
    assert pool.live_slots is not first


def test_reused_slot_initializes_all_follower_state() -> None:
    store = SynapseStore("edge", 1, 1, capacity=1, max_capacity=1)
    moment = DummyMomentColumn()
    store.followers().subscribe(moment)
    store.apply([birth("edge", [101])])
    first = store.view()
    slot = store._slots.slots_of(first.ids)

    store.age.tick()
    moment.values[slot] = 7.0
    assert store.age.values[slot].item() == 1
    assert store.lineage.values[slot].item() == 101

    store.apply([SynapseDeath("edge", first.ids), birth("edge", [202])])
    second = store.view()
    reused = store._slots.slots_of(second.ids)

    assert reused.tolist() == slot.tolist()
    assert second.ids.item() != first.ids.item()
    assert store.age.values[reused].item() == 0
    assert store.lineage.values[reused].item() == 202
    assert moment.values[reused].item() == 0.0
