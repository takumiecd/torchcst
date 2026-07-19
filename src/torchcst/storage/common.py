"""torchcst.storage.common — 共通機構 (継承ではなくコンポジションで使う)。

自作 store はこれらを使っても使わなくてもよい。EntityStore 契約さえ守れば。
"""

from __future__ import annotations

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


class SlotPool:
    """slot 割当・空き管理・id↔slot 解決・live_slots (派生キャッシュ;
    version 変化時のみ再構築)・compact remap の生成。

    v0 スコープ: capacity は固定 (拡張は未実装)。空きが尽きたら
    RuntimeError("capacity exhausted (growth not implemented in v0)")。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._id_to_slot: dict[int, int] = {}
        self._slot_to_id = torch.full((capacity,), -1, dtype=torch.int64)
        # 空き slot は昇順に維持する (allocate の決定性のため)。
        self._free_slots: list[int] = list(range(capacity))
        self._live_slots_cache: Tensor | None = None
        self._cache_valid = False

    @property
    def k_live(self) -> int:
        return self.capacity - len(self._free_slots)

    @property
    def live_slots(self) -> Tensor:
        """昇順 slot の packed LongTensor。version (allocate/release/compact)
        が動かない限りキャッシュを再利用する。"""
        if not self._cache_valid:
            self._live_slots_cache = torch.nonzero(
                self._slot_to_id >= 0, as_tuple=False
            ).flatten()
            self._cache_valid = True
        return self._live_slots_cache

    def allocate(self, ids: Tensor) -> Tensor:
        """ids に対応する新規 slot を割り当てて返す ([n])。
        バッチ一括 (Python ループは mutation 内部の帳簿付けのみ、forward
        経路からは呼ばれない)。"""
        n = int(ids.numel())
        if n > len(self._free_slots):
            raise RuntimeError(
                "capacity exhausted (growth not implemented in v0)"
            )
        chosen = self._free_slots[:n]
        self._free_slots = self._free_slots[n:]
        slots = torch.tensor(chosen, dtype=torch.int64)
        ids_list = ids.tolist()
        for slot, id_ in zip(chosen, ids_list):
            self._id_to_slot[id_] = slot
        self._slot_to_id[slots] = ids.to(torch.int64)
        self._cache_valid = False
        return slots

    def release(self, slots: Tensor) -> None:
        slots_list = slots.tolist()
        for slot in slots_list:
            id_ = int(self._slot_to_id[slot].item())
            if id_ < 0:
                raise KeyError(f"slot {slot} is already free")
            del self._id_to_slot[id_]
        self._slot_to_id[slots] = -1
        self._free_slots.extend(slots_list)
        self._free_slots.sort()
        self._cache_valid = False

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
