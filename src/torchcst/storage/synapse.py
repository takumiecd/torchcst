"""torchcst.storage.synapse — synapse column (store / view / ops)。

entity 縦割り: 実データ・mutation 意味論・op 語彙はこのモジュールで完結。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from ..contracts import EntityStore, Op, View
from .common import FollowerHub, IdAllocator, SlotPool


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


class SynapseStore(EntityStore):
    """ν = Σ w_k δ_(s_k,t_k)。実データ: s, t, w (全て学習対象)。

    mutation 意味論はここで完結:
      - birth: op が座標と初期 w を運ぶ。moment/計器は FollowerHub 経由で追従
      - death: 論理削除 (SlotPool.release + notify)
      - merge: s,t = w 質量重み平均 / w = 和。mass は自分の w から取る
      - kick : 座標への in-place 加算
    kernel の per-atom パラメタ (per-atom σ 等) は extras 列として
    add_extra() で追加し、merge 則を登録させる。
    """

    def __init__(self, site: str, d_in: int, d_out: int,
                 capacity: int, rank: int = 0, device=None):
        self.site = site
        self.d_in = d_in
        self.d_out = d_out
        self.capacity = capacity

        self._ids = IdAllocator(rank)
        self._slots = SlotPool(capacity)
        self._hub = FollowerHub()

        self.s = nn.Parameter(torch.zeros(capacity, d_in, device=device))
        self.t = nn.Parameter(torch.zeros(capacity, d_out, device=device))
        self.w = nn.Parameter(torch.zeros(capacity, device=device))

        self._version = 0
        # view() の gather index (live_slots) だけを version キーでキャッシュ
        # する。gather 結果そのもの (s/t/w の切り出し) は autograd グラフに
        # 入るため毎回作り直す — キャッシュするのは index tensor のみ。
        self._view_slots_cache_version = -1
        self._view_slots_cache: Tensor | None = None

    @property
    def version(self) -> int:
        return self._version

    def live_ids(self) -> Tensor:
        return self._slots.ids_of(self._slots.live_slots)

    def _live_slots_cached(self) -> Tensor:
        if self._view_slots_cache_version != self._version:
            self._view_slots_cache = self._slots.live_slots
            self._view_slots_cache_version = self._version
        return self._view_slots_cache

    def view(self) -> SynapseView:
        slots = self._live_slots_cached()
        ids = self._slots.ids_of(slots)
        s = self.s.index_select(0, slots)
        t = self.t.index_select(0, slots)
        w = self.w.index_select(0, slots)
        return SynapseView(site=self.site, version=self._version,
                            s=s, t=t, w=w, ids=ids)

    def apply(self, op: Op) -> None:
        """SynapseBirth/Death/Merge/Kick を受理。それ以外は TypeError。"""
        if isinstance(op, SynapseBirth):
            self._apply_birth(op)
        elif isinstance(op, SynapseDeath):
            self._apply_death(op)
        elif isinstance(op, (SynapseMerge, SynapseKick)):
            raise NotImplementedError("v0: merge/kick not implemented")
        else:
            raise TypeError(f"SynapseStore.apply: unsupported op type {type(op)!r}")

    def _apply_birth(self, op: SynapseBirth) -> None:
        if op.site != self.site:
            raise ValueError(
                f"SynapseBirth.site={op.site!r} does not match store site={self.site!r}"
            )
        n = op.s.shape[0]
        if op.t.shape[0] != n or op.w.shape[0] != n:
            raise ValueError("SynapseBirth: s, t, w must share leading dim (atom count)")
        if op.s.shape[-1] != self.d_in or op.t.shape[-1] != self.d_out:
            raise ValueError("SynapseBirth: s/t last dim must match d_in/d_out")

        ids = self._ids.issue(n)
        slots = self._slots.allocate(ids)
        with torch.no_grad():
            self.s.index_copy_(0, slots, op.s.to(device=self.s.device, dtype=self.s.dtype))
            self.t.index_copy_(0, slots, op.t.to(device=self.t.device, dtype=self.t.dtype))
            self.w.index_copy_(0, slots, op.w.to(device=self.w.device, dtype=self.w.dtype))
        self._hub.notify_birth(slots)
        self._version += 1

    def _apply_death(self, op: SynapseDeath) -> None:
        if op.site != self.site:
            raise ValueError(
                f"SynapseDeath.site={op.site!r} does not match store site={self.site!r}"
            )
        slots = self._slots.slots_of(op.ids)
        with torch.no_grad():
            self.s.index_copy_(0, slots, torch.zeros(slots.numel(), self.d_in,
                                                       device=self.s.device, dtype=self.s.dtype))
            self.t.index_copy_(0, slots, torch.zeros(slots.numel(), self.d_out,
                                                       device=self.t.device, dtype=self.t.dtype))
            self.w.index_copy_(0, slots, torch.zeros(slots.numel(),
                                                       device=self.w.device, dtype=self.w.dtype))
        self._slots.release(slots)
        self._hub.notify_death(slots)
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
