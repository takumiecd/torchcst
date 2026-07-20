"""slot配置と付随状態の共通機構。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch
from torch import Tensor


class IdAllocator:
    """int64 IDをnever-reuseな単調列から払い出す（上位bitは将来の用途に予約）。"""

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("start must be non-negative")
        self._next = start

    @property
    def next_id(self) -> int:
        """次回払い出すIDを返す。"""
        return self._next

    def preview(self, n: int) -> Tensor:
        """状態を変えずに次のID列を返す。"""
        self._validate_count(n)
        return torch.arange(self._next, self._next + n, dtype=torch.int64)

    def issue(self, n: int) -> Tensor:
        """次のID列を払い出してカウンタを進める。"""
        ids = self.preview(n)
        self._next += n
        return ids

    def _validate_count(self, n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an int")
        if n < 0:
            raise ValueError("n must be non-negative")
        if self._next + n > torch.iinfo(torch.int64).max:
            raise OverflowError("int64 entity ID space exhausted")


@dataclass(frozen=True)
class SlotBirth:
    """物理slotをcount個確保する要求。"""

    count: int


@dataclass(frozen=True)
class SlotDeath:
    """entity IDに対応する物理slotを解放する要求。"""

    ids: Tensor


SlotOp = SlotBirth | SlotDeath


@dataclass(frozen=True)
class SlotChange:
    """slot batch適用後の物理row差分。"""

    born_ids: Tensor
    born_slots: Tensor
    dead_ids: Tensor
    dead_slots: Tensor
    old_capacity: int
    new_capacity: int


@dataclass(frozen=True)
class _SlotPlan:
    """mutation前に検証済みのslot変更計画。"""

    pool: SlotPool
    version: int
    change: SlotChange


class SlotPool:
    """never-reuse IDと再利用可能slotを分離し、容量を倍々に拡張する。"""

    def __init__(
        self,
        capacity: int,
        *,
        max_capacity: int | None = None,
        allocator: IdAllocator | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if max_capacity is not None and max_capacity < capacity:
            raise ValueError("max_capacity must be at least capacity")
        self.capacity = capacity
        self.max_capacity = max_capacity
        self._allocator = allocator if allocator is not None else IdAllocator()
        self._id_to_slot: dict[int, int] = {}
        self._slot_to_id = torch.full((capacity,), -1, dtype=torch.int64)
        self._version = 0
        self._live_slots_cache: Tensor | None = None
        self._live_slots_cache_version = -1

    @property
    def version(self) -> int:
        """slot配置のversionを返す。"""
        return self._version

    @property
    def k_live(self) -> int:
        """生存entity数を返す。"""
        return len(self._id_to_slot)

    @property
    def live_slots(self) -> Tensor:
        """同一version中は再構築しない昇順live slot列を返す。"""
        if self._live_slots_cache_version != self._version:
            self._live_slots_cache = torch.nonzero(
                self._slot_to_id >= 0, as_tuple=False
            ).flatten()
            self._live_slots_cache_version = self._version
        assert self._live_slots_cache is not None
        return self._live_slots_cache

    def slots_of(self, ids: Tensor) -> Tensor:
        """生存entity IDを物理slotへ解決する。"""
        ids = self._validate_vector(ids, "ids")
        try:
            values = [self._id_to_slot[int(id_)] for id_ in ids.tolist()]
        except KeyError as exc:
            raise KeyError(f"unknown id: {exc.args[0]}") from None
        return torch.tensor(values, dtype=torch.int64)

    def ids_of(self, slots: Tensor) -> Tensor:
        """生存物理slotをentity IDへ解決する。"""
        slots = self._validate_vector(slots, "slots")
        invalid = (slots < 0) | (slots >= self.capacity)
        if bool(invalid.any()):
            raise KeyError(f"slots out of range: {slots[invalid].tolist()}")
        ids = self._slot_to_id.index_select(0, slots)
        if bool((ids < 0).any()):
            raise KeyError(f"slots not live: {slots[ids < 0].tolist()}")
        return ids

    def prepare(self, ops: Sequence[SlotOp]) -> _SlotPlan:
        """slot op列を一切mutationせず検証して計画する。"""
        n_birth, death_ids = self._normalize_ops(ops)
        death_slots = self.slots_of(death_ids)
        free = torch.nonzero(self._slot_to_id < 0, as_tuple=False).flatten().tolist()
        available = sorted(free + death_slots.tolist())
        required = self.k_live - death_ids.numel() + n_birth
        new_capacity = self._grown_capacity(required)
        available.extend(range(self.capacity, new_capacity))
        born_slots = torch.tensor(available[:n_birth], dtype=torch.int64)
        change = SlotChange(
            born_ids=self._allocator.preview(n_birth),
            born_slots=born_slots,
            dead_ids=death_ids,
            dead_slots=death_slots,
            old_capacity=self.capacity,
            new_capacity=new_capacity,
        )
        return _SlotPlan(pool=self, version=self._version, change=change)

    def commit(self, plan: _SlotPlan) -> SlotChange:
        """検証済みslot計画を現在versionへ一度だけ適用する。"""
        if plan.pool is not self:
            raise ValueError("slot plan belongs to another pool")
        if plan.version != self._version:
            raise RuntimeError("stale or already committed slot plan")
        change = plan.change
        issued = self._allocator.issue(change.born_ids.numel())
        if not torch.equal(issued, change.born_ids):
            raise RuntimeError("ID allocator changed after prepare")

        if change.new_capacity != self.capacity:
            grown = torch.full((change.new_capacity,), -1, dtype=torch.int64)
            grown[: self.capacity] = self._slot_to_id
            self._slot_to_id = grown
            self.capacity = change.new_capacity

        for id_ in change.dead_ids.tolist():
            del self._id_to_slot[id_]
        self._slot_to_id[change.dead_slots] = -1
        for slot, id_ in zip(change.born_slots.tolist(), issued.tolist()):
            self._id_to_slot[id_] = slot
        self._slot_to_id[change.born_slots] = issued
        self._version += 1
        return change

    def apply(self, ops: Sequence[SlotOp]) -> SlotChange:
        """slot op列をprepareして直ちにcommitする。"""
        return self.commit(self.prepare(ops))

    def _normalize_ops(self, ops: Sequence[SlotOp]) -> tuple[int, Tensor]:
        n_birth = 0
        deaths: list[Tensor] = []
        for op in ops:
            if isinstance(op, SlotBirth):
                if isinstance(op.count, bool) or not isinstance(op.count, int):
                    raise TypeError("SlotBirth.count must be an int")
                if op.count < 0:
                    raise ValueError("SlotBirth.count must be non-negative")
                n_birth += op.count
            elif isinstance(op, SlotDeath):
                ids = self._validate_vector(op.ids, "SlotDeath.ids")
                deaths.append(ids)
            else:
                raise TypeError(f"unsupported slot op type {type(op)!r}")
        death_ids = (
            torch.cat(deaths)
            if deaths
            else torch.zeros(0, dtype=torch.int64)
        )
        if death_ids.unique().numel() != death_ids.numel():
            raise ValueError("death ids must not contain duplicates")
        return n_birth, death_ids

    def _grown_capacity(self, required: int) -> int:
        new_capacity = self.capacity
        if required > new_capacity and new_capacity == 0:
            new_capacity = 1
        while required > new_capacity:
            new_capacity *= 2
        if self.max_capacity is not None and new_capacity > self.max_capacity:
            raise RuntimeError("capacity exhausted")
        return new_capacity

    @staticmethod
    def _validate_vector(value: Tensor, name: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank 1")
        if value.dtype != torch.int64:
            raise TypeError(f"{name} must have dtype int64")
        return value.detach().to(device="cpu").clone()


class Follower(Protocol):
    """再利用slotをdeathでclearしbirthで初期化する付随状態の契約。"""

    def grow(self, new_capacity: int) -> None: ...

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None: ...

    def on_death(self, slots: Tensor) -> None: ...

    def on_remap(self, old_to_new: Tensor) -> None: ...


class FollowerHub:
    """grow・birth・death・remapを全followerへ同順序で通知する。"""

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self._followers: list[Follower] = []

    def subscribe(self, follower: Follower) -> None:
        """followerを現在容量へgrowして購読登録する。"""
        follower.grow(self.capacity)
        self._followers.append(follower)

    def notify_grow(self, new_capacity: int) -> None:
        """全followerを新容量へ拡張する。"""
        if new_capacity < self.capacity:
            raise ValueError("FollowerHub cannot shrink")
        if new_capacity == self.capacity:
            return
        for follower in self._followers:
            follower.grow(new_capacity)
        self.capacity = new_capacity

    def notify_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """全followerのbirth slotを必ず初期化する。"""
        if slots.numel() != lineage.numel():
            raise ValueError("birth slots and lineage must have equal length")
        for follower in self._followers:
            follower.on_birth(slots, lineage)

    def notify_death(self, slots: Tensor) -> None:
        """全followerのdeath slotを必ずclearする。"""
        for follower in self._followers:
            follower.on_death(slots)

    def notify_remap(self, old_to_new: Tensor) -> None:
        """全followerへold-to-new slot写像を通知する。"""
        for follower in self._followers:
            follower.on_remap(old_to_new)


class AgeColumn:
    """birth後の構造event数をslotごとに保持する標準follower。"""

    def __init__(
        self,
        capacity: int = 0,
        live_slots: Callable[[], Tensor] | None = None,
    ) -> None:
        self.values = torch.zeros(capacity, dtype=torch.int64)
        self._live_slots = live_slots

    def grow(self, new_capacity: int) -> None:
        """既存ageを保ったまま列を拡張する。"""
        if new_capacity < self.values.numel():
            raise ValueError("AgeColumn cannot shrink")
        if new_capacity == self.values.numel():
            return
        grown = torch.zeros(new_capacity, dtype=torch.int64)
        grown[: self.values.numel()] = self.values
        self.values = grown

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """birth slotのageを0へ初期化する。"""
        self.values[slots] = 0

    def on_death(self, slots: Tensor) -> None:
        """death slotのageをclearする。"""
        self.values[slots] = 0

    def on_remap(self, old_to_new: Tensor) -> None:
        """生存slotのageを写像先へ移す。"""
        remapped = torch.zeros_like(self.values)
        old = torch.nonzero(old_to_new >= 0, as_tuple=False).flatten()
        remapped[old_to_new[old]] = self.values[old]
        self.values = remapped

    def tick(self, live_slots: Tensor | None = None) -> None:
        """live slotだけのageを1増やす。"""
        slots = live_slots
        if slots is None:
            if self._live_slots is None:
                raise ValueError("live_slots must be provided")
            slots = self._live_slots()
        self.values[slots] += 1


class LineageColumn:
    """birth op由来のint64 lineage keyをslotごとに保持する標準follower。"""

    def __init__(self, capacity: int = 0) -> None:
        self.values = torch.full((capacity,), -1, dtype=torch.int64)

    def grow(self, new_capacity: int) -> None:
        """既存lineageを保ったまま列を拡張する。"""
        if new_capacity < self.values.numel():
            raise ValueError("LineageColumn cannot shrink")
        if new_capacity == self.values.numel():
            return
        grown = torch.full((new_capacity,), -1, dtype=torch.int64)
        grown[: self.values.numel()] = self.values
        self.values = grown

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """birth slotへlineage keyを設定する。"""
        self.values[slots] = lineage

    def on_death(self, slots: Tensor) -> None:
        """death slotのlineageをclearする。"""
        self.values[slots] = -1

    def on_remap(self, old_to_new: Tensor) -> None:
        """生存slotのlineageを写像先へ移す。"""
        remapped = torch.full_like(self.values, -1)
        old = torch.nonzero(old_to_new >= 0, as_tuple=False).flatten()
        remapped[old_to_new[old]] = self.values[old]
        self.values = remapped
