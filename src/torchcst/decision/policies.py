"""torchcst.decision.policies — 参照 policy。

「各 1 画面で書けるか」が API の受け入れテスト。decide は全 site・
全 entity の読み値を見て混成 op バッチを返してよい (entity 独立ゲートを
強制しない)。純関数契約: 記録済み Reading でリプレイ可能・乱数は rng 経由。
"""

from __future__ import annotations

from typing import Callable

import torch

from ..contracts import Op, Reading, View
from ..storage.synapse import SynapseBirth, SynapseDeath
from .instruments import CandidateProbe, GradEMA


class cSET:
    """SET の CST 版: 質量最小 death + 一様ランダム座標 birth (w=0 init)。"""

    def __init__(self, sites: list[str], dt: int = 500, t_end: int = 50_000,
                 frac: Callable[[int], float] = lambda t: 0.3 * (1 - t / 50_000)):
        self.sites, self.dt, self.t_end, self.frac = sites, dt, t_end, frac

    def instruments(self):
        return {f"gema:{s}": (GradEMA(), s) for s in self.sites}

    def schedule(self, step):
        return ["death", "birth"] if step % self.dt == 0 and step < self.t_end else []

    def decide(self, phase, readings, views, rng):
        ops: list[Op] = []
        for site in self.sites:
            n = ...  # frac(step) * k_live
            if phase == "death":
                ops.append(SynapseDeath(site, readings[f"gema:{site}"].topk(n, largest=False)))
            else:
                s, t = ...  # rng で domain 一様サンプル
                ops.append(SynapseBirth(site, s=s, t=t, w=torch.zeros(n)))
        return ops


class cRigL:
    """RigL の CST 版: death は cSET 同様、birth は勾配場最大の候補座標。
    採択後に座標が off-grid で磨かれるのが離散版に無い CST の利点。"""

    def __init__(self, sites: list[str], dt: int = 500, t_end: int = 50_000,
                 pool: int = 4096):
        self.sites, self.dt, self.t_end, self.pool = sites, dt, t_end, pool

    def instruments(self):
        out = {}
        for s in self.sites:
            out[f"gema:{s}"] = (GradEMA(), s)
            out[f"probe:{s}"] = (CandidateProbe(pool=self.pool), s)
        return out

    def schedule(self, step):
        return ["death", "birth"] if step % self.dt == 0 and step < self.t_end else []

    def decide(self, phase, readings, views, rng):
        ops: list[Op] = []
        for site in self.sites:
            n = ...
            if phase == "death":
                ops.append(SynapseDeath(site, readings[f"gema:{site}"].topk(n, largest=False)))
            else:
                cand = readings[f"probe:{site}"]   # 座標付き Reading
                s, t = ...                          # cand.topk(n) の座標
                ops.append(SynapseBirth(site, s=s, t=t, w=torch.zeros(n)))
        return ops
