"""torchcst.decision.policies — 参照 policy。

「各 1 画面で書けるか」が API の受け入れテスト。schedule された判断処理は
全 site・全 entity の読み値を見て混成 op バッチを返してよい (entity 独立
ゲートを強制しない)。記録済み DecisionContext でリプレイ可能・乱数は
context の rng 経由のみ。
"""

from __future__ import annotations

from typing import Callable, cast

import torch

from ..contracts import DecisionContext, DecisionStage, Op
from ..storage.synapse import SynapseBirth, SynapseDeath
from .instruments import CandidateProbe, CandidateReading, GradEMA, MassEMA


class cSET:
    """SET の CST 版: 質量最小 death + 一様ランダム座標 birth (w=0 init)。

    死亡判定は MassEMA (|w| の EMA) を使う — GradEMA は v0 未実装
    (instruments.py の「計器の kernel アクセス問題」参照) なのでこちらで
    代替する。

    rewire は変更前の同一スナップショットから death と birth を一括決定し、
    SynapseDeath → SynapseBirth の順で返す。これにより原子数を保ったまま、
    phase 間で pending 状態を持たずに済む。
    """

    def __init__(self, sites: list[str], dt: int = 500, t_end: int = 50_000,
                 frac: Callable[[int], float] = lambda t: 0.3 * (1 - t / 50_000),
                 domain: tuple[float, float] = (0.0, 1.0)):
        self.sites, self.dt, self.t_end, self.frac = sites, dt, t_end, frac
        self.domain = domain

    def instruments(self):
        return {f"mass:{s}": (MassEMA(), s) for s in self.sites}

    def schedule(self, step: int) -> list[DecisionStage]:
        if step % self.dt == 0 and step < self.t_end:
            return [DecisionStage("rewire", tuple(self.sites), self.rewire)]
        return []

    def rewire(self, site: str, ctx: DecisionContext) -> list[Op]:
        lo, hi = self.domain
        view = ctx.views[site]
        k_live = int(view.ids.numel())
        n = min(max(1, int(self.frac(ctx.step) * k_live)), k_live)
        if n == 0:
            return []

        dying = ctx.readings[f"mass:{site}"].topk(n, largest=False)
        d_in = view.s.shape[-1]
        d_out = view.t.shape[-1]
        s = lo + (hi - lo) * torch.rand(n, d_in, generator=ctx.rng)
        t = lo + (hi - lo) * torch.rand(n, d_out, generator=ctx.rng)
        return [
            SynapseDeath(site, ids=dying),
            SynapseBirth(site, s=s, t=t, w=torch.zeros(n)),
        ]


class cRigL:
    """RigL の CST 版: death は質量最小 (MassEMA、cSET と同じ判定)。birth は
    勾配場最大の候補座標 (CandidateProbe、KernelPort 経由で実装済み)。
    採択後に座標が off-grid で磨かれるのが離散版に無い CST の利点。

    death が MassEMA なのは原典への忠実な対応: RigL (Evci et al. 2020) の
    drop 基準は weight magnitude 最小 (SET と同じ) であり、gradient
    magnitude を使うのは grow 側だけ。grad ベースの死亡判定を試したい
    policy は GradEMA (実装済み) に差し替えればよい — 死亡判定の instrument
    は Policy が自由に選べる。

    cSET と同様、rewire は変更前の同一スナップショットから death と birth
    を一括決定し、順序付きの op バッチを返す。

    GateEMA / RentCounter はまだ未実装 (instruments.py 参照: GateEMA は
    ∂L/∂c に pre-gate 値が要るという KernelPort だけでは解決しない別問題)
    なので、この policy では使わない。
    """

    def __init__(self, sites: list[str], dt: int = 500, t_end: int = 50_000,
                 pool: int = 4096,
                 frac: Callable[[int], float] = lambda t: 0.3 * (1 - t / 50_000),
                 domain: tuple[float, float] = (0.0, 1.0)):
        self.sites, self.dt, self.t_end, self.pool = sites, dt, t_end, pool
        self.frac = frac
        self.domain = domain

    def instruments(self):
        out = {}
        for s in self.sites:
            out[f"mass:{s}"] = (MassEMA(), s)
            out[f"probe:{s}"] = (CandidateProbe(pool=self.pool, domain=self.domain), s)
        return out

    def schedule(self, step: int) -> list[DecisionStage]:
        if step % self.dt == 0 and step < self.t_end:
            return [DecisionStage("rewire", tuple(self.sites), self.rewire)]
        return []

    def rewire(self, site: str, ctx: DecisionContext) -> list[Op]:
        view = ctx.views[site]
        k_live = int(view.ids.numel())
        n = min(max(1, int(self.frac(ctx.step) * k_live)), k_live)
        if n == 0:
            return []

        dying = ctx.readings[f"mass:{site}"].topk(n, largest=False)
        probe = cast(CandidateReading, ctx.readings[f"probe:{site}"])
        s, t = probe.topk_coords(n)
        return [
            SynapseDeath(site, ids=dying),
            SynapseBirth(site, s=s, t=t, w=torch.zeros(n)),
        ]
