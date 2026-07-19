"""torchcst.storage.synapse — synapse column (store / view / ops)。

entity 縦割り: 実データ・mutation 意味論・op 語彙はこのモジュールで完結。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

from .base import EntityStore, Op, View
from .common import (
    FollowerHub,
    SlotBackend,
    SlotBirth,
    SlotChange,
    SlotDeath,
    SlotOp,
    SlotPool,
)


@dataclass(frozen=True)
class SynapseView(View):
    s: Tensor        # [K_live, d_in]
    t: Tensor        # [K_live, d_out]
    w: Tensor        # [K_live]
    ids: Tensor      # [K_live] int64
    extras: dict[str, Tensor] = field(default_factory=dict)  # per-atom σ など


@dataclass(frozen=True)
class SynapseBirth:
    site: str
    s: Tensor
    t: Tensor
    w: Tensor        # 通常 zeros (RigL 流)


@dataclass(frozen=True)
class SynapseDeath:
    site: str
    ids: Tensor


@dataclass(frozen=True)
class SynapseMerge:
    site: str
    id_pairs: Tensor  # [n, 2]


@dataclass(frozen=True)
class SynapseKick:
    """座標への明示摂動 (探索ノイズ / 位置キック)。"""

    site: str
    ids: Tensor
    ds: Tensor | None = None
    dt: Tensor | None = None


@dataclass(frozen=True)
class _SynapseBatch:
    """Entity Op列を検証・正規化した、SynapseStore内部のmutation値。"""

    slot_ops: tuple[SlotOp, ...]
    s: Tensor
    t: Tensor
    w: Tensor

    @classmethod
    def from_ops(
        cls, store: "SynapseStore", ops: Sequence[Op]
    ) -> "_SynapseBatch":
        births: list[SynapseBirth] = []
        slot_ops: list[SlotOp] = []
        for op in ops:
            if isinstance(op, (SynapseMerge, SynapseKick)):
                raise NotImplementedError("v0: merge/kick not implemented")
            if not isinstance(op, (SynapseBirth, SynapseDeath)):
                raise TypeError(
                    f"SynapseStore.apply: unsupported op type {type(op)!r}"
                )
            if op.site != store.site:
                raise ValueError(
                    f"op.site={op.site!r} does not match store site={store.site!r}"
                )

            if isinstance(op, SynapseDeath):
                slot_ops.append(SlotDeath(op.ids))
                continue

            if op.s.ndim != 2 or op.t.ndim != 2 or op.w.ndim != 1:
                raise ValueError("SynapseBirth: expected s/t rank 2 and w rank 1")
            n = op.w.shape[0]
            if op.s.shape[0] != n or op.t.shape[0] != n:
                raise ValueError(
                    "SynapseBirth: s, t, w must share leading dim (atom count)"
                )
            if op.s.shape[-1] != store.d_in or op.t.shape[-1] != store.d_out:
                raise ValueError("SynapseBirth: s/t last dim must match d_in/d_out")
            births.append(op)
            slot_ops.append(SlotBirth(n))

        if not births:
            return cls(
                slot_ops=tuple(slot_ops),
                s=store.s.new_zeros((0, store.d_in)),
                t=store.t.new_zeros((0, store.d_out)),
                w=store.w.new_zeros((0,)),
            )
        return cls(
            slot_ops=tuple(slot_ops),
            s=torch.cat([op.s for op in births]).to(store.s),
            t=torch.cat([op.t for op in births]).to(store.t),
            w=torch.cat([op.w for op in births]).to(store.w),
        )

    def write(self, store: "SynapseStore", change: SlotChange) -> None:
        with torch.no_grad():
            store.s.index_fill_(0, change.dead_slots, 0.0)
            store.t.index_fill_(0, change.dead_slots, 0.0)
            store.w.index_fill_(0, change.dead_slots, 0.0)
            store.s.index_copy_(0, change.born_slots, self.s)
            store.t.index_copy_(0, change.born_slots, self.t)
            store.w.index_copy_(0, change.born_slots, self.w)


class SynapseStore(EntityStore):
    """ν = Σ w_k δ_(s_k,t_k)。実データ: s, t, w (全て学習対象)。

    mutation 意味論はここで完結:
      - birth: opが座標と初期wを運ぶ。slot-indexed付随状態はFollowerHubで追従
      - birth/death: batch全体をSlotBackendで同時に計画・検証してから適用
      - merge: s,t = w 質量重み平均 / w = 和。mass は自分の w から取る
      - kick : 座標への in-place 加算
    kernel の per-atom パラメタ (per-atom σ 等) は extras 列として
    add_extra() で追加し、merge 則を登録させる。
    """

    def __init__(self, site: str, d_in: int, d_out: int,
                 capacity: int, rank: int = 0, device=None,
                 slot_pool: SlotBackend | None = None):
        self.site = site
        self.d_in = d_in
        self.d_out = d_out
        self.capacity = capacity

        self._slots = slot_pool if slot_pool is not None else SlotPool(capacity, rank)
        if self._slots.capacity != capacity:
            raise ValueError("slot_pool.capacity must match store capacity")
        self._hub = FollowerHub()

        self.s = nn.Parameter(torch.zeros(capacity, d_in, device=device))
        self.t = nn.Parameter(torch.zeros(capacity, d_out, device=device))
        self.w = nn.Parameter(torch.zeros(capacity, device=device))

        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def live_ids(self) -> Tensor:
        return self._slots.ids_of(self._slots.live_slots)

    def view(self) -> SynapseView:
        slots = self._slots.live_slots
        ids = self._slots.ids_of(slots)
        s = self.s.index_select(0, slots)
        t = self.t.index_select(0, slots)
        w = self.w.index_select(0, slots)
        return SynapseView(site=self.site, version=self._version,
                            capacity=self.capacity,
                            s=s, t=t, w=w, ids=ids)

    def apply(self, ops: Sequence[Op]) -> None:
        """site-local な op batch を全件検証してから一括適用する。

        version はopごとでなく、空でないbatch全体につき一度だけ進める。
        StoreはOp型・site・tensor shapeだけを検証し、ID生存性・capacity・
        cardinalityと物理row配置はSlotBackendへ委譲する。
        """
        ops = list(ops)
        if not ops:
            return
        batch = _SynapseBatch.from_ops(self, ops)
        change = self._slots.apply(batch.slot_ops)
        batch.write(self, change)

        if change.dead_slots.numel():
            self._hub.notify_death(change.dead_slots)
        if change.born_slots.numel():
            self._hub.notify_birth(change.born_slots)
        self._version += 1

    def parameters(self) -> Iterable[nn.Parameter]:
        return [self.s, self.t, self.w]

    def followers(self) -> FollowerHub:
        return self._hub

    def canonical_state(self) -> dict:
        """P5 相当だが v0 は compact (物理詰め直し) をせず、id 順に並べた
        {s,t,w,ids} を返すだけに留める。"""
        view = self.view()
        order = torch.argsort(view.ids)
        return {
            "ids": view.ids.index_select(0, order).detach().clone(),
            "s": view.s.index_select(0, order).detach().clone(),
            "t": view.t.index_select(0, order).detach().clone(),
            "w": view.w.index_select(0, order).detach().clone(),
        }

    def load_state(self, state: dict) -> None:
        raise NotImplementedError("v0: SynapseStore.load_state not implemented")

    def add_extra(self, name: str, shape: tuple[int, ...],
                  merge: Callable[[Tensor, Tensor], Tensor],
                  learnable: bool = False) -> None:
        raise NotImplementedError("v0: SynapseStore.add_extra not implemented")
