"""torchcst.contracts — 契約面のすべてがこのファイルに集約される。

三角形:
    Storage ──View──→ Compute ──Observation──→ Decision ──Op──→ Storage

頂点間の会話はこの 3 型 (View / Observation / Op) 以外で行わない:
  - Compute は Op を発行できない (構造を変えられない)
  - Decision は View の読みと Op の発行のみ (重みテンソルに触れない)
  - Engine は Op の中身を見ない (site でルーティングして apply するだけ)

entity 縦割り: 具体型 (SynapseView / NeuronBirth / ...) は各 store の
モジュールに置かれる。ここにあるのは共通契約だけ。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .storage.common import FollowerHub
    from .storage.synapse import SynapseStore

# ── 辺の型 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class View:
    """Storage → Compute / Decision。live 原子の packed 読み取りビュー。
    entity ごとにサブクラスで具体化する (dict 渡しにしない)。"""

    site: str
    version: int


@dataclass(frozen=True)
class Observation:
    """Compute → Decision。backward の瞬間に site で観測できる生データ。

    x と g_out から dense 勾配 G = g_outᵀ x が非実体化のまま導けるのが
    存在理由 (「まだ無い原子の勾配」は .grad に無い)。hook は detach して
    詰める以上のことをしない。"""

    site: str
    x: Tensor        # 層入力 [B, N_in]
    g_out: Tensor    # 出力勾配 [B, N_out]
    version: int     # 観測時の structure_version (跨いだ EMA の無効化判断用)


class Op(Protocol):
    """mutation 命令の共通形。中身は entity ごとに自由に定義してよい。
    Engine は site でルーティングし、broadcast のため serialize 可能で
    あることだけ要求する (P4)。"""

    site: str


# ── Storage 契約 ───────────────────────────────────────────────────────

class Follower(Protocol):
    """FollowerHub の購読契約 (moment 列・計器状態が実装する)。
    意味論 (birth 初期値・merge 合成則) は follower 側が決める。"""

    def grow(self, new_capacity: int) -> None: ...
    def on_birth(self, slots: Tensor) -> None: ...
    def on_death(self, slots: Tensor) -> None: ...
    def on_merge(self, src_slots: Tensor, dst_slots: Tensor,
                 mass: Tensor) -> None: ...
    def on_remap(self, old_to_new: Tensor) -> None: ...


class EntityStore(ABC):
    """Engine が store について知るすべて。内部実装は自由。

    契約 4 条:
      - 保持テンソルの leading dim は capacity と常に一致
      - apply は同期的トランザクション (適用 → version++。拒否・遅延不可、
        不正 op は例外で落とす)
      - slot 順に意味を持たせない。永続参照は id (P1)
      - mutation/フック内で per-item 同期をしない (バッチ一括のみ)
    """

    site: str

    @property
    @abstractmethod
    def version(self) -> int: ...
    @abstractmethod
    def live_ids(self) -> Tensor: ...
    @abstractmethod
    def view(self) -> View: ...
    @abstractmethod
    def apply(self, op: Op) -> None: ...
    @abstractmethod
    def parameters(self) -> Iterable[nn.Parameter]: ...
    @abstractmethod
    def followers(self) -> "FollowerHub":
        """moment/計器の追従登録口。Engine と Instrument がここに subscribe。"""
        ...
    @abstractmethod
    def canonical_state(self) -> dict: ...
    @abstractmethod
    def load_state(self, state: dict) -> None: ...


# ── Compute 契約 ───────────────────────────────────────────────────────

class Kernel(Protocol):
    """κ の族。global パラメタは自前 nn.Parameter。per-atom パラメタが
    必要なら install(store) で SynapseStore.add_extra を呼んで列を確保する。
    パラメタ不要な kernel は両方とも no-op。"""

    def install(self, store: "SynapseStore") -> None: ...
    def global_params(self) -> Iterable[nn.Parameter]: ...
    def __call__(self, query: Tensor, centers: Tensor,
                 extras: dict[str, Tensor]) -> Tensor:
        """κ(query_i − center_k) を評価して [N_query, K] を返す。"""
        ...


# ── Decision 契約 ──────────────────────────────────────────────────────

class Reading(Protocol):
    """decide() に渡す計器の読み値 (読み取り専用・id ベース)。"""

    def topk(self, k: int, largest: bool = True) -> Tensor: ...  # -> ids
    def value(self, ids: Tensor) -> Tensor: ...


class Instrument(ABC):
    """計器: Observation を毎 step 煮詰め、ΔT ごとの判断材料に変える蓄積器
    (mutation は ΔT ごと・観測は毎 step、の時間スケール差を埋める)。

    entity 縦割り: synapse 計器は SynapseStore に、neuron 計器は
    NeuronStore に bind する。per-atom 状態は Follower として
    store.followers() に subscribe → mutation 追従が自動 (P3)。

    契約: 入力は Observation と View のみ / obs.version が変わったら
    EMA 系は自分で無効化 / 答えは常に id で返す (slot を漏らさない)。"""

    @abstractmethod
    def bind(self, store: EntityStore) -> None: ...
    @abstractmethod
    def update(self, obs: Observation, view: View) -> None: ...
    @abstractmethod
    def read(self) -> Reading: ...


class Policy(Protocol):
    """自由度の唯一の置き場・横断点。

    decide は「全 site・全 entity の読み値」を見て混成 op バッチ
    (例 [NeuronDeath(...), SynapseMerge(...)]) を返せる。entity 独立の
    ゲートに分解することを API は強制しない (SC-MRG-1 / SC-LIFE-2 の教訓:
    独立ゲートは半端均衡・necessary-atom 食いの病理を生む)。
    純関数契約: 記録済み Reading でリプレイ可能・乱数は rng 経由のみ (P4)。"""

    def instruments(self) -> dict[str, tuple[Instrument, str]]:
        """{計器名: (instance, bind先 site)}。Engine が bind と配線を行う。"""
        ...

    def schedule(self, step: int) -> list[str]:
        """発火フェーズ名を実行順で (例 ["death", "birth"])。空 = 何もしない。"""
        ...

    def decide(self, phase: str, readings: dict[str, Reading],
               views: dict[str, View], rng: torch.Generator) -> list[Op]: ...
