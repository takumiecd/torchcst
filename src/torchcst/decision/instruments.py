"""torchcst.decision.instruments — 同梱計器 (初期 3 policy が要る分だけ)。

計器は entity 縦割り: synapse 計器は SynapseStore に、neuron 計器は
NeuronStore に bind する。per-atom 状態は Follower として
store.followers() に subscribe → mutation 追従が自動 (P3)。

「計器の kernel アクセス問題」の裁定 (このアークで解決): GradEMA が本来
見るべき ∂L/∂w_k は Observation(x, g_out) だけからは計算できず、κ_in/κ_out
の評価 (= kernel そのもの) が要る。Observation に kernel を同梱する案は
却下した (contracts.py の KernelPort docstring 参照 — Observation は生データ、
kernel は意味論なので per-step の辺を汚したくない)。代わりに Engine が
Instrument.bind() 時に KernelPort (読み取り専用の評価能力) を渡す方式を
採用した。GradEMA / CandidateProbe はこの port 経由で実装済み。

GateEMA / RentCounter は本アークでもまだ未実装: GateEMA が要る ∂L/∂c は
KernelPort の in_features/out_features (w 側の合成) だけでは出てこない
別の設計問題 (gate 自体は κ の外側で x/y に直接掛かる pre-gate 値が要る)
なので、次アーク以降の課題として残す。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..contracts import EntityStore, Instrument, KernelPort, Observation, Reading, View


@dataclass(frozen=True)
class IdReading:
    """id ベースの Reading 実装。ids[i] に対応する values[i] を保持する
    (読み取り専用・スナップショット)。"""

    ids: Tensor
    values: Tensor

    def topk(self, k: int, largest: bool = True) -> Tensor:
        k = max(0, min(k, int(self.ids.numel())))
        if k == 0:
            return self.ids.new_zeros(0)
        _, idx = torch.topk(self.values, k, largest=largest)
        return self.ids.index_select(0, idx)

    def value(self, ids: Tensor) -> Tensor:
        # id -> value のバッチ一括ルックアップ (per-id ループ禁止)。
        if self.ids.numel() == 0:
            return torch.full((ids.numel(),), float("nan"))
        match = ids.unsqueeze(1) == self.ids.unsqueeze(0)   # [n_query, n_have]
        found = match.any(dim=1)
        idx = match.float().argmax(dim=1)
        out = self.values.index_select(0, idx)
        return torch.where(found, out, torch.full_like(out, float("nan")))


class _IdSpaceEMA:
    """id 空間で per-atom EMA を保持する共通ヘルパ (MassEMA / GradEMA で
    共用 — reconcile ロジックの二重化を避けるためここに一度だけ書く)。

    per-atom 状態は **id 空間** で保持する (FollowerHub の slot-indexed
    buffer 方式は使わない: view が slot を外部に漏らさない P1 と、この
    ヘルパが id ベースでしか答えを返さない契約に合わせるため)。

    reconcile は version が前回から変わった時だけ行う: 同一 version 内では
    view の並び順 (SlotPool.live_slots 昇順) が安定しているため、毎 update
    で id マッチングをやり直す必要はない。reconcile 自体は torch.isin ベース
    のバッチ一括処理で、Python の per-id ループはしない。
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.version = -1
        self.ids = torch.zeros(0, dtype=torch.int64)
        self.ema = torch.zeros(0)

    def reconcile(self, new_ids: Tensor, new_version: int) -> None:
        if self.ids.numel() and new_ids.numel():
            survive_mask_old = torch.isin(self.ids, new_ids)
            old_ids_survive = self.ids[survive_mask_old]
            old_ema_survive = self.ema[survive_mask_old]
        else:
            old_ids_survive = self.ids.new_zeros(0)
            old_ema_survive = self.ema.new_zeros(0)

        new_ema = torch.zeros(new_ids.numel(), dtype=self.ema.dtype
                               if self.ema.numel() else torch.get_default_dtype())
        if old_ids_survive.numel():
            # old_ids_survive の各 id が new_ids のどこにいるかをバッチで求める
            match = old_ids_survive.unsqueeze(1) == new_ids.unsqueeze(0)
            pos = match.float().argmax(dim=1)
            new_ema[pos] = old_ema_survive

        self.ids = new_ids
        self.ema = new_ema
        self.version = new_version

    def update(self, new_ids: Tensor, new_version: int, sample: Tensor) -> None:
        if new_version != self.version:
            self.reconcile(new_ids, new_version)
        self.ema = self.decay * self.ema + (1.0 - self.decay) * sample

    def read(self) -> "IdReading":
        return IdReading(ids=self.ids, values=self.ema)


class MassEMA(Instrument):
    """|w| の EMA (synapse 用)。cSET の死亡候補判定はこれで完結する —
    Observation を実質使わず view.w だけで作れるので KernelPort は不要
    (bind 時に渡されても無視する)。
    """

    def __init__(self, decay: float = 0.9):
        self._core = _IdSpaceEMA(decay)
        self._store: EntityStore | None = None

    def bind(self, store: EntityStore, port: KernelPort | None = None) -> None:
        self._store = store  # port は使わない (view だけで完結する計器)

    def update(self, obs: Observation, view: View) -> None:
        if not hasattr(view, "w"):
            raise TypeError("MassEMA requires a SynapseView (view.w missing)")
        self._core.update(view.ids, view.version, view.w.detach().abs())

    def read(self) -> Reading:
        return self._core.read()


class GradEMA(Instrument):
    """|∂L/∂w| の EMA (synapse 用)。死亡候補の定番。

    KernelPort 経由で

        grad_w = (port.in_features(obs.x, view.s)
                  * port.out_features(obs.g_out, view.t)).sum(dim=0)

    を評価する (gate 込み・no_grad・バッチ一括 — 導出は contracts.KernelPort
    の docstring 参照)。per-atom 状態は MassEMA と同じ _IdSpaceEMA を使う。

    port が None で bind されたら (= synapse site が CSTLinear に属さない
    異常な配線) 使う段になって落ちるより早く、bind 時点で即エラーにする。
    """

    def __init__(self, decay: float = 0.9):
        self._core = _IdSpaceEMA(decay)
        self._store: EntityStore | None = None
        self._port: KernelPort | None = None

    def bind(self, store: EntityStore, port: KernelPort | None = None) -> None:
        if port is None:
            raise ValueError(
                "GradEMA requires a KernelPort — bind site must be a synapse "
                "site owned by a CSTLinear (got port=None)"
            )
        self._store = store
        self._port = port

    def update(self, obs: Observation, view: View) -> None:
        if not hasattr(view, "w"):
            raise TypeError("GradEMA requires a SynapseView (view.s/t missing)")
        with torch.no_grad():
            f_in = self._port.in_features(obs.x, view.s)        # [B, K]
            f_out = self._port.out_features(obs.g_out, view.t)  # [B, K]
            grad_w = (f_in * f_out).sum(dim=0).abs()
        self._core.update(view.ids, view.version, grad_w)

    def read(self) -> Reading:
        return self._core.read()


class GateEMA(Instrument):
    """|c| / |∂L/∂c| の EMA (gated neuron 用)。neuron death の判断材料 —
    v0 ではまだ未実装: KernelPort の in_features/out_features は w 側の
    合成 (Σ_k ... ) しか出さないので、gate 自体への勾配 ∂L/∂c (κ の外側で
    x/y に直接掛かる pre-gate 値が要る) はこの port では復元できない。
    別の port 設計が要る、という形で課題を残す。"""

    def bind(self, store: EntityStore, port: KernelPort | None = None) -> None:
        raise NotImplementedError(
            "v0: GateEMA not implemented — needs a different port exposing "
            "pre-gate x/y (KernelPort's in/out_features only give the "
            "w-side composition, not d L / d c)"
        )

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError("v0: GateEMA not implemented")

    def read(self) -> Reading:
        raise NotImplementedError("v0: GateEMA not implemented")


class RentCounter(Instrument):
    """rent gate 用の効用/家賃カウンタ (RENT policy の本体)。両 entity 対応
    — v0 では未実装 (rent の効用定義そのものが本アークのスコープ外)。"""

    def bind(self, store: EntityStore, port: KernelPort | None = None) -> None:
        raise NotImplementedError("v0: RentCounter not implemented")

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError("v0: RentCounter not implemented")

    def read(self) -> Reading:
        raise NotImplementedError("v0: RentCounter not implemented")


@dataclass(frozen=True)
class CandidateReading:
    """CandidateProbe.read() の戻り値。まだ生まれていない候補原子には id が
    無い (IdAllocator は SynapseStore.apply 時にしか払い出さない) ので、
    Reading (id ベース) とは意図的に別物にしてある — topk/value の代わりに
    座標そのものを返す topk_coords(k) を提供する。"""

    s: Tensor        # [P, d_in]
    t: Tensor        # [P, d_out]
    scores: Tensor   # [P]

    def topk_coords(self, k: int) -> tuple[Tensor, Tensor]:
        k = max(0, min(k, int(self.scores.numel())))
        if k == 0:
            return self.s.new_zeros(0, self.s.shape[-1]), self.t.new_zeros(0, self.t.shape[-1])
        _, idx = torch.topk(self.scores, k, largest=True)
        return self.s.index_select(0, idx), self.t.index_select(0, idx)


class CandidateProbe(Instrument):
    """birth 候補の off-support 勾配場プローブ (synapse 用; cRigL の本体)。

    候補座標プール (s_c [P,d_in], t_c [P,d_out]) は domain 上の一様サンプル
    (proposal="uniform" のみ v0 対応)。毎 update で KernelPort 経由の

        score_c += | (port.in_features(x, s_c) * port.out_features(g_out, t_c)).sum(0) |

    を dense G 非実体化のまま一括評価して EMA する (GradEMA と全く同じ
    3 行式 — 対象が live atom の s/t でなく候補プールの s_c/t_c なだけ)。

    **view.version が変わったら pool を再サンプルし score を 0 リセットす
    る**: mutation が起きた直後は候補プールの座標に対する旧 score が stale
    になる (死んだ原子跡地に立っていた候補のスコアが高いまま残る、あるいは
    前回 topk で採用され実際に birth 済みの候補がまだ pool に居座って
    「二重採用」される、といった問題を機械的に断ち切るため)。

    候補プールは派生データであり checkpoint 対象外 (P5 のスコープ外)。
    """

    def __init__(self, pool: int = 4096, proposal: str = "uniform",
                 domain: tuple[float, float] = (0.0, 1.0),
                 decay: float = 0.9, seed: int = 0):
        if proposal != "uniform":
            raise NotImplementedError(
                f"v0: CandidateProbe proposal={proposal!r} not implemented "
                "— only 'uniform' is supported"
            )
        self.pool = pool
        self.proposal = proposal
        self.domain = domain
        self.decay = decay
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        self._store: EntityStore | None = None
        self._port: KernelPort | None = None
        self._version = -1
        self._s: Tensor | None = None
        self._t: Tensor | None = None
        self._scores: Tensor | None = None

    def bind(self, store: EntityStore, port: KernelPort | None = None) -> None:
        if port is None:
            raise ValueError(
                "CandidateProbe requires a KernelPort — bind site must be a "
                "synapse site owned by a CSTLinear (got port=None)"
            )
        self._store = store
        self._port = port

    def _resample(self, d_in: int, d_out: int) -> None:
        lo, hi = self.domain
        self._s = lo + (hi - lo) * torch.rand(self.pool, d_in, generator=self._rng)
        self._t = lo + (hi - lo) * torch.rand(self.pool, d_out, generator=self._rng)
        self._scores = torch.zeros(self.pool)

    def update(self, obs: Observation, view: View) -> None:
        if not hasattr(view, "w"):
            raise TypeError("CandidateProbe requires a SynapseView")

        # d_in/d_out は最初の update 時の view.s/view.t から lazy に取る
        # (bind 時点では store がまだ空でも構わない設計)。
        if self._s is None or view.version != self._version:
            d_in = view.s.shape[-1]
            d_out = view.t.shape[-1]
            self._resample(d_in, d_out)
            self._version = view.version

        with torch.no_grad():
            f_in = self._port.in_features(obs.x, self._s)          # [B, P]
            f_out = self._port.out_features(obs.g_out, self._t)    # [B, P]
            score_batch = (f_in * f_out).sum(dim=0).abs()
        self._scores = self.decay * self._scores + (1.0 - self.decay) * score_batch

    def read(self) -> CandidateReading:
        return CandidateReading(s=self._s, t=self._t, scores=self._scores)
