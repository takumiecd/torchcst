"""torchcst.decision.instruments — 同梱計器 (初期 3 policy が要る分だけ)。

計器は entity 縦割り: synapse 計器は SynapseStore に、neuron 計器は
NeuronStore に bind する。per-atom 状態は Follower として
store.followers() に subscribe → mutation 追従が自動 (P3)。
"""

from __future__ import annotations

from ..contracts import EntityStore, Instrument, Observation, Reading, View


class GradEMA(Instrument):
    """|∂L/∂w| の EMA (synapse 用)。死亡候補の定番。"""

    def bind(self, store: EntityStore) -> None: ...
    def update(self, obs: Observation, view: View) -> None: ...
    def read(self) -> Reading: ...


class GateEMA(Instrument):
    """|c| / |∂L/∂c| の EMA (gated neuron 用)。neuron death の判断材料。"""

    def bind(self, store: EntityStore) -> None: ...
    def update(self, obs: Observation, view: View) -> None: ...
    def read(self) -> Reading: ...


class RentCounter(Instrument):
    """rent gate 用の効用/家賃カウンタ (RENT policy の本体)。両 entity 対応。"""

    def bind(self, store: EntityStore) -> None: ...
    def update(self, obs: Observation, view: View) -> None: ...
    def read(self) -> Reading: ...


class CandidateProbe(Instrument):
    """birth 候補の off-support 勾配場プローブ (synapse 用; cRigL の本体)。

    候補座標プール (派生データ; checkpoint 対象外) を保持し、毎 update で
      score_c += | (g_outᵀ κ_out(t_c)) · (κ_in(s_c)ᵀ x) |
    を dense G 非実体化で一括評価して EMA。read() は座標付き Reading。
    proposal="uniform" | "near_support" | callable。
    """

    def __init__(self, pool: int = 4096, proposal="uniform"): ...

    def bind(self, store: EntityStore) -> None: ...
    def update(self, obs: Observation, view: View) -> None: ...
    def read(self) -> Reading: ...
