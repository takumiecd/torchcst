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
from .instruments import CandidateProbe, GradEMA, MassEMA


class cSET:
    """SET の CST 版: 質量最小 death + 一様ランダム座標 birth (w=0 init)。

    死亡判定は MassEMA (|w| の EMA) を使う — GradEMA は v0 未実装
    (instruments.py の「計器の kernel アクセス問題」参照) なのでこちらで
    代替する。

    妥協 2 点 (docstring に明記):
      1. Policy.decide には step が渡ってこない設計 (Policy.decide の
         シグネチャに step 引数がない) ため、schedule(step) で self._step
         に保存して decide から参照する。
      2. birth の原子数は death と同数にしたい (k_live を維持する) が、
         death で k_live が減った後の view から再計算すると値がずれうる
         ので、death phase で計算した n を self._pending_n[site] に控えて
         birth phase で使い回す (schedule() が必ず ["death", "birth"] の
         順で返すことに依存する内部実装)。
    """

    def __init__(self, sites: list[str], dt: int = 500, t_end: int = 50_000,
                 frac: Callable[[int], float] = lambda t: 0.3 * (1 - t / 50_000),
                 domain: tuple[float, float] = (0.0, 1.0)):
        self.sites, self.dt, self.t_end, self.frac = sites, dt, t_end, frac
        self.domain = domain
        self._step = 0
        self._pending_n: dict[str, int] = {}

    def instruments(self):
        return {f"mass:{s}": (MassEMA(), s) for s in self.sites}

    def schedule(self, step):
        self._step = step
        return ["death", "birth"] if step % self.dt == 0 and step < self.t_end else []

    def decide(self, phase, readings, views, rng):
        ops: list[Op] = []
        lo, hi = self.domain
        for site in self.sites:
            view = views[site]
            if phase == "death":
                k_live = int(view.ids.numel())
                n = max(1, int(self.frac(self._step) * k_live))
                n = min(n, k_live)
                self._pending_n[site] = n
                dying = readings[f"mass:{site}"].topk(n, largest=False)
                ops.append(SynapseDeath(site, ids=dying))
            else:  # phase == "birth"
                n = self._pending_n.get(site, 0)
                if n <= 0:
                    continue
                d_in = view.s.shape[-1]
                d_out = view.t.shape[-1]
                s = lo + (hi - lo) * torch.rand(n, d_in, generator=rng)
                t = lo + (hi - lo) * torch.rand(n, d_out, generator=rng)
                ops.append(SynapseBirth(site, s=s, t=t, w=torch.zeros(n)))
        return ops


class cRigL:
    """RigL の CST 版: death は cSET 同様、birth は勾配場最大の候補座標。
    採択後に座標が off-grid で磨かれるのが離散版に無い CST の利点。

    v0 では未着手のまま (CandidateProbe / GradEMA が「計器の kernel
    アクセス問題」で未実装のため — instruments.py 参照)。"""

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
