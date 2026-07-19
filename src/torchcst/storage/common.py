"""torchcst.storage.common — 共通機構 (継承ではなくコンポジションで使う)。

自作 store はこれらを使っても使わなくてもよい。EntityStore 契約さえ守れば。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor

from ..contracts import Follower

# id の上位ビットに詰める rank のビット幅 (P2)。counter はこの下位に収まる
# 前提 (rank ごとの発行数が 2**48 を超えることは v0 では想定しない)。
_RANK_SHIFT = 48


class IdAllocator:
    """P2 の id 払い出し器。int64、never-reuse、単調増加、上位ビットに rank
    (分散時の birth を無調停にする)。"""

    def __init__(self, rank: int = 0):
        self._rank = rank
        self._next = 0

    def issue(self, n: int) -> Tensor:
        """int64 [n]。呼ぶたびに単調増加するカウンタから払い出す
        (release されても再利用しない = never-reuse)。"""
        start = self._next
        ids = torch.arange(start, start + n, dtype=torch.int64)
        self._next += n
        if self._rank:
            ids = ids | (self._rank << _RANK_SHIFT)
        return ids


@dataclass(frozen=True)
class SlotBirth:
    count: int


@dataclass(frozen=True)
class SlotDeath:
    ids: Tensor


SlotOp = SlotBirth | SlotDeath


@dataclass(frozen=True)
class _SlotBatch:
    """SlotOp列を正規化した、検証済みのbirth/death要求。"""

    n_birth: int
    death_ids: Tensor

    @classmethod
    def from_ops(cls, ops: Sequence[SlotOp]) -> "_SlotBatch":
        n_birth = 0
        deaths: list[Tensor] = []
        for op in ops:
            if isinstance(op, SlotBirth):
                if op.count < 0:
                    raise ValueError("SlotBirth.count must be non-negative")
                n_birth += op.count
            elif isinstance(op, SlotDeath):
                if op.ids.ndim != 1:
                    raise ValueError("SlotDeath.ids must be rank 1")
                deaths.append(op.ids)
            else:
                raise TypeError(f"unsupported slot op type {type(op)!r}")

        death_ids = (
            torch.cat(deaths).to(torch.int64)
            if deaths else torch.zeros(0, dtype=torch.int64)
        )
        if death_ids.unique().numel() != death_ids.numel():
            raise ValueError("death ids must not contain duplicates")
        return cls(n_birth=n_birth, death_ids=death_ids)


@dataclass(frozen=True)
class SlotChange:
    """SlotPool.apply()が返す、entity tensor更新用の物理row差分。"""

    born_ids: Tensor
    born_slots: Tensor
    dead_ids: Tensor
    dead_slots: Tensor


class SlotBackend(Protocol):
    """EntityStoreがslot配置について依存する最小契約。"""

    capacity: int

    @property
    def k_live(self) -> int: ...
    @property
    def live_slots(self) -> Tensor: ...
    def apply(self, ops: Sequence[SlotOp]) -> SlotChange: ...
    def slots_of(self, ids: Tensor) -> Tensor: ...
    def ids_of(self, slots: Tensor) -> Tensor: ...


class SlotPool:
    """SlotBirth/SlotDeath batchの検証・ID発行・物理row更新を一括実行する。

    あわせて空き管理・id↔slot解決・live_slots cache・compact remapを持つ。
    EntityStoreはslotの生存性やcapacityを再検証せず、apply()へ委譲する。

    v0 スコープ: capacity は固定 (拡張は未実装)。空きが尽きたら
    RuntimeError("capacity exhausted (growth not implemented in v0)")。
    """

    def __init__(self, capacity: int, rank: int = 0):
        self.capacity = capacity
        self._ids = IdAllocator(rank)
        self._id_to_slot: dict[int, int] = {}
        self._slot_to_id = torch.full((capacity,), -1, dtype=torch.int64)
        # 空き slot は昇順に維持する (birth 配置の決定性のため)。
        self._free_slots: list[int] = list(range(capacity))
        self._live_slots_cache: Tensor | None = None
        self._cache_valid = False

    @property
    def k_live(self) -> int:
        return self.capacity - len(self._free_slots)

    @property
    def live_slots(self) -> Tensor:
        """昇順 slot の packed LongTensor。apply/compact されるまで再利用する。"""
        if not self._cache_valid:
            self._live_slots_cache = torch.nonzero(
                self._slot_to_id >= 0, as_tuple=False
            ).flatten()
            self._cache_valid = True
        return self._live_slots_cache

    def apply(self, ops: Sequence[SlotOp]) -> SlotChange:
        """slot op batchを全件検証後、一回のmutationとして適用する。"""
        batch = _SlotBatch.from_ops(ops)
        death_slots = self.slots_of(batch.death_ids)
        available = sorted(self._free_slots + death_slots.tolist())
        if batch.n_birth > len(available):
            raise RuntimeError(
                "capacity exhausted (growth not implemented in v0)"
            )

        birth_slots = torch.tensor(available[:batch.n_birth], dtype=torch.int64)
        birth_ids = self._ids.issue(batch.n_birth)
        for id_ in batch.death_ids.tolist():
            del self._id_to_slot[id_]
        self._slot_to_id[death_slots] = -1

        for slot, id_ in zip(birth_slots.tolist(), birth_ids.tolist()):
            self._id_to_slot[id_] = slot
        self._slot_to_id[birth_slots] = birth_ids

        self._free_slots = torch.nonzero(
            self._slot_to_id < 0, as_tuple=False
        ).flatten().tolist()
        self._cache_valid = False
        return SlotChange(
            born_ids=birth_ids,
            born_slots=birth_slots,
            dead_ids=batch.death_ids,
            dead_slots=death_slots,
        )

    def slots_of(self, ids: Tensor) -> Tensor:
        try:
            slots = [self._id_to_slot[int(i)] for i in ids.tolist()]
        except KeyError as exc:
            raise KeyError(f"unknown id: {exc.args[0]}") from None
        return torch.tensor(slots, dtype=torch.int64)

    def ids_of(self, slots: Tensor) -> Tensor:
        # forward 経路 (view) からも呼ばれるためバッチ一括で。
        slots = slots.to(torch.int64)
        ids = self._slot_to_id.index_select(0, slots)
        if bool((ids < 0).any()):
            bad = slots[ids < 0].tolist()
            raise KeyError(f"slots not live: {bad}")
        return ids

    def compact(self) -> Tensor:
        """live slot を昇順に前詰めする。old→new remap ([capacity] int64;
        死んでいた slot は -1) を返し、内部の帳簿もその配置に合わせて
        書き換える (id↔slot の対応そのものは不変、位置だけが変わる)。"""
        live = self.live_slots
        k = int(live.numel())
        new_positions = torch.arange(k, dtype=torch.int64)

        remap = torch.full((self.capacity,), -1, dtype=torch.int64)
        remap[live] = new_positions

        ids_at_live = self._slot_to_id.index_select(0, live)
        new_slot_to_id = torch.full((self.capacity,), -1, dtype=torch.int64)
        new_slot_to_id[new_positions] = ids_at_live
        self._slot_to_id = new_slot_to_id
        self._id_to_slot = {
            int(id_): int(slot)
            for id_, slot in zip(ids_at_live.tolist(), new_positions.tolist())
        }
        self._free_slots = list(range(k, self.capacity))
        self._cache_valid = False
        return remap


class BalancedSlotPool:
    """固定個数の置換に特化した、省メモリなSlotBackend実装。

    初回birthでKを確定した後はlive slotを常に0..K-1に保ち、deathしたrowを
    同じbatchのbirthが再利用する。free list・capacity長のslot表・id辞書を
    永続保持せず、必要な帳簿はslot順のlive ID tensorだけ。

    ID→slotはmutation時にtensor検索する。通常のSlotPoolより検索コストを
    払う代わりに、永続メモリをO(K)のint64 tensor 1本に抑える。
    """

    def __init__(self, capacity: int, rank: int = 0):
        self.capacity = capacity
        self._ids = IdAllocator(rank)
        self._ids_by_slot = torch.zeros(0, dtype=torch.int64)

    @property
    def k_live(self) -> int:
        return self._ids_by_slot.numel()

    @property
    def live_slots(self) -> Tensor:
        return torch.arange(self.k_live, dtype=torch.int64)

    def apply(self, ops: Sequence[SlotOp]) -> SlotChange:
        batch = _SlotBatch.from_ops(ops)
        n_death = batch.death_ids.numel()

        if self.k_live == 0:
            if n_death:
                raise KeyError(f"unknown id: {int(batch.death_ids[0])}")
            if batch.n_birth > self.capacity:
                raise RuntimeError(
                    "capacity exhausted (growth not implemented in v0)"
                )
            born_slots = torch.arange(batch.n_birth, dtype=torch.int64)
            born_ids = self._ids.issue(batch.n_birth)
            self._ids_by_slot = born_ids.clone()
            return SlotChange(
                born_ids=born_ids,
                born_slots=born_slots,
                dead_ids=batch.death_ids,
                dead_slots=torch.zeros(0, dtype=torch.int64),
            )

        if batch.n_birth != n_death:
            raise ValueError(
                "BalancedSlotPool requires equal birth and death counts "
                f"(birth={batch.n_birth}, death={n_death})"
            )

        dead_slots = self.slots_of(batch.death_ids)
        born_slots = dead_slots.sort().values
        born_ids = self._ids.issue(batch.n_birth)
        self._ids_by_slot[born_slots] = born_ids
        return SlotChange(
            born_ids=born_ids,
            born_slots=born_slots,
            dead_ids=batch.death_ids,
            dead_slots=dead_slots,
        )

    def slots_of(self, ids: Tensor) -> Tensor:
        ids = ids.to(device="cpu", dtype=torch.int64)
        if ids.ndim != 1:
            raise ValueError("ids must be rank 1")
        if not ids.numel():
            return torch.zeros(0, dtype=torch.int64)

        sorted_ids, slots = self._ids_by_slot.sort()
        positions = torch.searchsorted(sorted_ids, ids)
        in_range = positions < sorted_ids.numel()
        probe = positions.clamp_max(sorted_ids.numel() - 1)
        found = in_range & (sorted_ids.index_select(0, probe) == ids)
        if not bool(found.all()):
            unknown = int(ids[~found][0])
            raise KeyError(f"unknown id: {unknown}")
        return slots.index_select(0, positions)

    def ids_of(self, slots: Tensor) -> Tensor:
        slots = slots.to(device="cpu", dtype=torch.int64)
        if slots.ndim != 1:
            raise ValueError("slots must be rank 1")
        if bool(((slots < 0) | (slots >= self.k_live)).any()):
            bad = slots[(slots < 0) | (slots >= self.k_live)].tolist()
            raise KeyError(f"slots not live: {bad}")
        return self._ids_by_slot.index_select(0, slots)

    def compact(self) -> Tensor:
        """常にpackedなので、live slotへの恒等remapを返す。"""
        remap = torch.full((self.capacity,), -1, dtype=torch.int64)
        remap[:self.k_live] = self.live_slots
        return remap


class FollowerHub:
    """P3 の追従機構。moment 列・計器列など「本体に付随する slot-indexed
    配列」を購読させ、mutation 時に一括通知する。意味論 (birth 初期値や
    merge 合成則) は follower 側が決める。機構はこの 1 クラスで共通。"""

    def __init__(self):
        self._followers: list[Follower] = []

    def subscribe(self, follower: Follower) -> None:
        self._followers.append(follower)

    def notify_grow(self, new_capacity: int) -> None:
        for f in self._followers:
            f.grow(new_capacity)

    def notify_birth(self, slots: Tensor) -> None:
        for f in self._followers:
            f.on_birth(slots)

    def notify_death(self, slots: Tensor) -> None:
        for f in self._followers:
            f.on_death(slots)

    def notify_merge(self, src_slots: Tensor, dst_slots: Tensor,
                     mass: Tensor) -> None:
        for f in self._followers:
            f.on_merge(src_slots, dst_slots, mass)

    def notify_remap(self, old_to_new: Tensor) -> None:
        for f in self._followers:
            f.on_remap(old_to_new)
