"""torchcst.storage.synapse — synapse column (store / view / ops)。

entity 縦割り: 実データ・mutation 意味論・op 語彙はこのモジュールで完結。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from torch import Tensor

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
        # self._ids = IdAllocator(rank); self._slots = SlotPool(capacity)
        # self._hub = FollowerHub(); s/t/w は nn.Parameter [capacity, ...]
        ...

    def view(self) -> SynapseView: ...

    def apply(self, op: Op) -> None:
        """SynapseBirth/Death/Merge/Kick を受理。それ以外は TypeError。"""
        ...

    def add_extra(self, name: str, shape: tuple[int, ...],
                  merge: Callable[[Tensor, Tensor], Tensor],
                  learnable: bool = False) -> None: ...
