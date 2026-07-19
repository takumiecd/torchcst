"""torchcst.storage.neuron — neuron column (store / view / ops)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from ..contracts import EntityStore, Op, View
from .common import FollowerHub, IdAllocator


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
                 rank: int = 0):
        if learnable_coords:
            raise NotImplementedError("v0: learnable_coords=True not implemented")

        self.site = site
        self.gated = gated
        # v0: 標本点は固定集合 (mutation 非対応)。学習しないので buffer 相当
        # (plain Tensor・requires_grad なし) として保持する。
        self.mu = coords.detach().clone()
        self.mu.requires_grad_(False)

        n = self.mu.shape[0]
        self.c = nn.Parameter(torch.ones(n)) if gated else None

        self._ids_alloc = IdAllocator(rank)
        self._ids = self._ids_alloc.issue(n)
        self._hub = FollowerHub()

    @property
    def version(self) -> int:
        # v0: mutation 未実装のため固定集合 = version は常に 0。
        return 0

    def live_ids(self) -> Tensor:
        return self._ids

    def view(self) -> NeuronView:
        gate = self.c if self.gated else None
        return NeuronView(site=self.site, version=self.version,
                           mu=self.mu, gate=gate, ids=self._ids)

    def apply(self, ops: Sequence[Op]) -> None:
        """NeuronBirth/Death/Kick を受理。"""
        if not ops:
            return
        raise NotImplementedError(
            "v0: NeuronStore.apply (birth/death/kick) not implemented — "
            "標本点は固定集合として扱う"
        )

    def parameters(self) -> Iterable[nn.Parameter]:
        return [self.c] if self.gated else []

    def followers(self) -> FollowerHub:
        return self._hub

    def canonical_state(self) -> dict:
        raise NotImplementedError("v0: NeuronStore.canonical_state not implemented")

    def load_state(self, state: dict) -> None:
        raise NotImplementedError("v0: NeuronStore.load_state not implemented")
