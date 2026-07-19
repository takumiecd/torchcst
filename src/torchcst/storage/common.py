"""torchcst.storage.common — 共通機構 (継承ではなくコンポジションで使う)。

自作 store はこれらを使っても使わなくてもよい。EntityStore 契約さえ守れば。
"""

from __future__ import annotations

from torch import Tensor

from ..contracts import Follower


class IdAllocator:
    """P2 の id 払い出し器。int64、never-reuse、単調増加、上位ビットに rank
    (分散時の birth を無調停にする)。"""

    def __init__(self, rank: int = 0): ...
    def issue(self, n: int) -> Tensor: ...          # int64 [n]


class SlotPool:
    """slot 割当・空き管理・capacity 倍々拡張・id↔slot 解決・live_slots
    (派生キャッシュ; version 変化時のみ再構築)・compact remap の生成。"""

    def __init__(self, capacity: int): ...

    capacity: int

    @property
    def k_live(self) -> int: ...
    @property
    def live_slots(self) -> Tensor: ...
    def allocate(self, ids: Tensor) -> Tensor: ...  # -> slots [n] (拡張要求含む)
    def release(self, slots: Tensor) -> None: ...
    def slots_of(self, ids: Tensor) -> Tensor: ...
    def ids_of(self, slots: Tensor) -> Tensor: ...
    def compact(self) -> Tensor: ...                # old→new remap


class FollowerHub:
    """P3 の追従機構。moment 列・計器列など「本体に付随する slot-indexed
    配列」を購読させ、mutation 時に一括通知する。意味論 (birth 初期値や
    merge 合成則) は follower 側が決める。機構はこの 1 クラスで共通。"""

    def subscribe(self, follower: Follower) -> None: ...
    def notify_grow(self, new_capacity: int) -> None: ...
    def notify_birth(self, slots: Tensor) -> None: ...
    def notify_death(self, slots: Tensor) -> None: ...
    def notify_merge(self, src_slots: Tensor, dst_slots: Tensor,
                     mass: Tensor) -> None: ...
    def notify_remap(self, old_to_new: Tensor) -> None: ...
