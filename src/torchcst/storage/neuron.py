"""torchcst.storage.neuron — neuron column (store / view / ops)。"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..contracts import EntityStore, Op, View
from .common import FollowerHub, IdAllocator, SlotPool


@dataclass(frozen=True)
class NeuronView(View):
    mu: Tensor              # [N_live, d]
    gate: Tensor | None     # [N_live] (gated のみ; plain は None)
    ids: Tensor


@dataclass(frozen=True)
class NeuronBirth:
    site: str
    mu: Tensor
    gate: Tensor | None = None   # gated store で None なら初期値規約に従う


@dataclass(frozen=True)
class NeuronDeath:
    site: str
    ids: Tensor


@dataclass(frozen=True)
class NeuronKick:
    site: str
    ids: Tensor
    dmu: Tensor


class NeuronStore(EntityStore):
    """ニューロン標本点。plain (mu のみ) / gated (mu + 強度 c) の 2 種で
    打ち止め。l 層 dst と l+1 層 src の座標分離は「別 store を使う」という
    合成で表す (種類は増やさない)。

    neuron 固有の意味論 (synapse と共通化しない部分):
      - gated の c は BN gamma 系の連続振幅。death 判定の主対象
      - neuron merge は現時点では提供しない (必要になった時に op を足す —
        op 語彙は entity ローカルなので後方互換で追加できる)
    """

    def __init__(self, site: str, coords: Tensor, *, gated: bool = False,
                 learnable_coords: bool = False, capacity: int | None = None,
                 rank: int = 0): ...

    def view(self) -> NeuronView: ...

    def apply(self, op: Op) -> None:
        """NeuronBirth/Death/Kick を受理。"""
        ...
