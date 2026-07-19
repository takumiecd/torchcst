"""torchcst.decision.instruments — 同梱計器 (初期 3 policy が要る分だけ)。

計器は entity 縦割り: synapse 計器は SynapseStore に、neuron 計器は
NeuronStore に bind する。per-atom 状態は Follower として
store.followers() に subscribe → mutation 追従が自動 (P3)。

設計上の未解決点 (v0 で明示的に残す): GradEMA が本来見るべき ∂L/∂w_k は
Observation(x, g_out) だけからは計算できない。w_k に対する勾配を復元するには
κ_in(μ_i^in − s_k) / κ_out(μ_j^out − t_k) の評価 (= kernel そのもの) が要る
が、Instrument.update(obs, view) には kernel オブジェクトへのアクセス経路が
存在しない (Kernel は Compute 層である CSTLinear にしか渡っていない —
Instrument/Policy は Storage の View しか見ない三角形の外側にいる)。この
経路をどう設計するか (例: Kernel を Observation に同梱する / Instrument.bind
に kernel も渡せる形にする) は次アーク以降の課題として残す。GradEMA /
GateEMA / RentCounter / CandidateProbe はこの理由で v0 では
NotImplementedError のままにしておく。

その代わり v0 の cSET は **MassEMA** (|w| の EMA) で完結させる —
view だけで作れる計器なので上記の未解決問題を踏まない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..contracts import EntityStore, Instrument, Observation, Reading, View


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


class MassEMA(Instrument):
    """|w| の EMA (synapse 用)。cSET の死亡候補判定はこれで完結する —
    Observation を実質使わず view.w だけで作れるので、上記の kernel アクセス
    問題を踏まない。

    per-atom 状態は **id 空間** で保持する (FollowerHub の slot-indexed
    buffer 方式は使わない: view が slot を外部に漏らさない P1 と、この
    instrument が id ベースでしか答えを返さない契約に合わせるため。
    slot-indexed buffer は「今どの slot が誰か」を追う仕組みなので、
    id 空間で直接持つ方針とは重複投資になる)。

    reconcile は view.version が前回から変わった時だけ行う: 同一 version
    内では view の並び順 (SlotPool.live_slots 昇順) が安定しているため、
    毎 update で id マッチングをやり直す必要はない。reconcile 自体は
    torch.isin ベースのバッチ一括処理で、Python の per-id ループはしない。
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self._store: EntityStore | None = None
        self._version = -1
        self._ids = torch.zeros(0, dtype=torch.int64)
        self._ema = torch.zeros(0)

    def bind(self, store: EntityStore) -> None:
        self._store = store

    def _reconcile(self, view: View) -> None:
        new_ids = view.ids
        if self._ids.numel() and new_ids.numel():
            survive_mask_old = torch.isin(self._ids, new_ids)
            old_ids_survive = self._ids[survive_mask_old]
            old_ema_survive = self._ema[survive_mask_old]
        else:
            old_ids_survive = self._ids.new_zeros(0)
            old_ema_survive = self._ema.new_zeros(0)

        new_ema = torch.zeros(new_ids.numel(), dtype=self._ema.dtype
                               if self._ema.numel() else torch.get_default_dtype())
        if old_ids_survive.numel():
            # old_ids_survive の各 id が new_ids のどこにいるかをバッチで求める
            match = old_ids_survive.unsqueeze(1) == new_ids.unsqueeze(0)
            pos = match.float().argmax(dim=1)
            new_ema[pos] = old_ema_survive

        self._ids = new_ids
        self._ema = new_ema
        self._version = view.version

    def update(self, obs: Observation, view: View) -> None:
        if not hasattr(view, "w"):
            raise TypeError("MassEMA requires a SynapseView (view.w missing)")
        if view.version != self._version:
            self._reconcile(view)
        mass = view.w.detach().abs()
        self._ema = self.decay * self._ema + (1.0 - self.decay) * mass

    def read(self) -> Reading:
        return IdReading(ids=self._ids, values=self._ema)


class GradEMA(Instrument):
    """|∂L/∂w| の EMA (synapse 用)。死亡候補の定番 — だが v0 では未実装:
    上記モジュール docstring の「計器の kernel アクセス問題」参照。"""

    def bind(self, store: EntityStore) -> None:
        raise NotImplementedError(
            "v0: GradEMA not implemented — needs kernel access design "
            "(dL/dw_k requires kappa_in/kappa_out, unreachable from "
            "Observation(x, g_out) alone)"
        )

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError(
            "v0: GradEMA not implemented — needs kernel access design"
        )

    def read(self) -> Reading:
        raise NotImplementedError(
            "v0: GradEMA not implemented — needs kernel access design"
        )


class GateEMA(Instrument):
    """|c| / |∂L/∂c| の EMA (gated neuron 用)。neuron death の判断材料 —
    v0 では未実装: ∂L/∂c 側も同じ kernel アクセス問題を抱える。"""

    def bind(self, store: EntityStore) -> None:
        raise NotImplementedError(
            "v0: GateEMA not implemented — needs kernel access design"
        )

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError(
            "v0: GateEMA not implemented — needs kernel access design"
        )

    def read(self) -> Reading:
        raise NotImplementedError(
            "v0: GateEMA not implemented — needs kernel access design"
        )


class RentCounter(Instrument):
    """rent gate 用の効用/家賃カウンタ (RENT policy の本体)。両 entity 対応
    — v0 では未実装 (効用計算に kernel アクセスが要る場合の設計が未解決)。"""

    def bind(self, store: EntityStore) -> None:
        raise NotImplementedError(
            "v0: RentCounter not implemented — needs kernel access design"
        )

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError(
            "v0: RentCounter not implemented — needs kernel access design"
        )

    def read(self) -> Reading:
        raise NotImplementedError(
            "v0: RentCounter not implemented — needs kernel access design"
        )


class CandidateProbe(Instrument):
    """birth 候補の off-support 勾配場プローブ (synapse 用; cRigL の本体)。

    候補座標プール (派生データ; checkpoint 対象外) を保持し、毎 update で
      score_c += | (g_outᵀ κ_out(t_c)) · (κ_in(s_c)ᵀ x) |
    を dense G 非実体化で一括評価して EMA。read() は座標付き Reading。
    proposal="uniform" | "near_support" | callable。

    v0 では未実装: score 式そのものが κ_in/κ_out の評価を要求しており、
    GradEMA と同じ kernel アクセス問題を抱える (cRigL は着手しない)。
    """

    def __init__(self, pool: int = 4096, proposal="uniform"):
        self.pool = pool
        self.proposal = proposal

    def bind(self, store: EntityStore) -> None:
        raise NotImplementedError(
            "v0: CandidateProbe not implemented — needs kernel access design"
        )

    def update(self, obs: Observation, view: View) -> None:
        raise NotImplementedError(
            "v0: CandidateProbe not implemented — needs kernel access design"
        )

    def read(self) -> Reading:
        raise NotImplementedError(
            "v0: CandidateProbe not implemented — needs kernel access design"
        )
