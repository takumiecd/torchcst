"""torchcst.engine — 三角形の配線と Executor (ユーザーが触る唯一の入口)。"""

from __future__ import annotations

import torch
from torch import nn

from .contracts import Op, Policy


class CSTEngine:
    """配線:
      - model から stores を収集 (site 重複は即エラー)
      - policy.instruments() を bind し、backward hook で Observation を配る
      - store.parameters() を optimizer に接続し、moment 影列を Follower
        として followers() に subscribe (P3)
      - step(): schedule 発火 → decide → op を site でルーティングし
        store.apply (P4: version++, op ログ, 派生キャッシュ無効化)
      - 分散: rank0 で decide → op broadcast → 全 rank 同一適用
    Engine は op の中身も store の内部レイアウトも知らない。
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 policy: Policy, *, rank: int = 0, world_size: int = 1): ...

    def step(self) -> list[Op]:
        """optimizer.step() の後に呼ぶ。適用 op を返す (ロギング用)。"""
        ...

    def op_log(self) -> list[tuple[int, Op]]: ...
    def save(self, path: str) -> None: ...   # P5 正準形
    def load(self, path: str) -> None: ...
