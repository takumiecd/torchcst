"""cstf — CST framework API draft v0.3  (2026-07-19)

v0.2 からの変更 (合意済み設計):
  - 汎用 AtomTable+Layout を廃止し、entity 縦割りへ。SynapseStore /
    NeuronStore が各自の実データ・mutation 実装・専用 op 型を持つ
    (内部は密結合・自由)。Engine が知るのは EntityStore 契約のみ。
  - 共通機構 (id/slot/follower) は継承でなくコンポジションで提供。
  - 計器も entity 縦割り。ただし Policy.decide と Compute (layer 合成) の
    2 点だけは横断を API 上保証する (SC-MRG-1 / SC-LIFE-2 の教訓:
    entity 独立ゲートは半端均衡・necessary-atom 食いの病理を再生産する)。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全体構造 = 三角形 × entity 縦割りプリズム:

               synapse column          neuron column
  Storage   │ SynapseStore + ops   │ NeuronStore + ops   │ ← 縦割り(自由)
  計器      │ synapse Instruments  │ neuron Instruments  │ ← 縦割り(自由)
  ──────────┼──────────────────────┴─────────────────────┤
  Policy    │   横断: 全計器の読み値 → 協調 op バッチ       │
  Compute   │   横断: CSTLinear = synapse × neuron の合成   │

  三角形 (v0.2 から不変):
        Storage ──View──→ Compute ──Observation──→ Decision ──Op──→ Storage
  辺の型はこの 3 つだけ。頂点間はこれ以外で会話しない:
    - Compute は Op を発行できない (構造を変えられない)
    - Decision は View の読みと Op の発行のみ (重みテンソルに触れない)
    - Engine は Op の中身を見ない (site でルーティングして apply するだけ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ストレージ第一原理 (確定事項・不変):
  P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
  P2. id は int64・never-reuse・単調増加・上位ビット rank (分散 birth 無調停)。
  P3. per-atom 付随状態 (moment/計器) は follower として mutation に自動追従。
  P4. mutation は store へのトランザクション。version 単調増加 + op ログ。
      分散は op ログの broadcast 同一適用 (テンソル同期なし)。
  P5. checkpoint は正準形 (compact → id ソート) でバイト決定的。
      派生データ (live_slots キャッシュ, 空間インデックス, 候補プール) は対象外。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import torch
from torch import Tensor, nn

# ══════════════════════════════════════════════════════════════════════
# 0. 共通機構 — 継承ではなくコンポジションで使う内部部品
#    (自作 store はこれらを使っても使わなくてもよい。契約さえ守れば)
# ══════════════════════════════════════════════════════════════════════

class IdAllocator:
    """P2 の id 払い出し器。int64、never-reuse、上位ビットに rank。"""

    def __init__(self, rank: int = 0): ...
    def issue(self, n: int) -> Tensor: ...          # int64 [n]


class SlotPool:
    """slot 割当・空き管理・capacity 倍々拡張・id↔slot 解決・live_slots
    (派生キャッシュ; version 変化時のみ再構築)・compact remap の生成。"""

    def __init__(self, capacity: int): ...
    capacity: int
    @property
    def k_live(self) -> int: ...
    @property
    def live_slots(self) -> Tensor: ...
    def allocate(self, ids: Tensor) -> Tensor: ...  # -> slots [n] (拡張要求含む)
    def release(self, slots: Tensor) -> None: ...
    def slots_of(self, ids: Tensor) -> Tensor: ...
    def ids_of(self, slots: Tensor) -> Tensor: ...
    def compact(self) -> Tensor: ...                # old→new remap


class FollowerHub:
    """P3 の追従機構。moment 列・計器列など「本体に付随する slot-indexed
    配列」を購読させ、mutation 時に一括通知する。意味論 (birth 初期値や
    merge 合成則) は follower 側が決める。機構はこの 1 クラスで共通。"""

    def subscribe(self, follower: "Follower") -> None: ...
    def notify_grow(self, new_capacity: int) -> None: ...
    def notify_birth(self, slots: Tensor) -> None: ...
    def notify_death(self, slots: Tensor) -> None: ...
    def notify_merge(self, src_slots: Tensor, dst_slots: Tensor,
                     mass: Tensor) -> None: ...
    def notify_remap(self, old_to_new: Tensor) -> None: ...


class Follower(Protocol):
    """FollowerHub の購読契約 (moment 列・計器状態が実装する)。"""

    def grow(self, new_capacity: int) -> None: ...
    def on_birth(self, slots: Tensor) -> None: ...
    def on_death(self, slots: Tensor) -> None: ...
    def on_merge(self, src_slots: Tensor, dst_slots: Tensor,
                 mass: Tensor) -> None: ...
    def on_remap(self, old_to_new: Tensor) -> None: ...


# ══════════════════════════════════════════════════════════════════════
# 1. Engine 契約 — Engine が store について知るすべて
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class View:
    """Storage → Compute / Decision。live 原子の packed 読み取りビュー。
    entity ごとにサブクラスで具体化する (dict 渡しにしない)。"""

    site: str
    version: int


class Op(Protocol):
    """mutation 命令の共通形。中身は entity ごとに自由に定義してよい。
    Engine は site でルーティングし、broadcast のため serialize 可能で
    あることだけ要求する (P4)。"""

    site: str


class EntityStore(ABC):
    """Engine が要求する最小契約。内部実装 (SoA/コンポジション/自作) は自由。

    契約 4 条 (v0.2 から不変):
      - 保持テンソルの leading dim は capacity と常に一致
      - apply は同期的トランザクション (適用 → version++ 。拒否・遅延不可、
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
    def followers(self) -> FollowerHub:
        """moment/計器の追従登録口。Engine と Instrument がここに subscribe。"""
        ...
    @abstractmethod
    def canonical_state(self) -> dict: ...
    @abstractmethod
    def load_state(self, state: dict) -> None: ...


# ══════════════════════════════════════════════════════════════════════
# 2. synapse column — store / view / ops (内部は密結合・自由)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SynapseView(View):
    s: Tensor        # [K_live, d_in]
    t: Tensor        # [K_live, d_out]
    w: Tensor        # [K_live]
    ids: Tensor      # [K_live] int64
    extras: dict[str, Tensor] = field(default_factory=dict)  # per-atom σ など


@dataclass(frozen=True)
class SynapseBirth:
    site: str
    s: Tensor
    t: Tensor
    w: Tensor        # 通常 zeros (RigL 流)

@dataclass(frozen=True)
class SynapseDeath:
    site: str
    ids: Tensor

@dataclass(frozen=True)
class SynapseMerge:
    site: str
    id_pairs: Tensor  # [n, 2]

@dataclass(frozen=True)
class SynapseKick:
    """座標への明示摂動 (探索ノイズ / 位置キック)。"""
    site: str
    ids: Tensor
    ds: Tensor | None = None
    dt: Tensor | None = None


class SynapseStore(EntityStore):
    """ν = Σ w_k δ_(s_k,t_k)。実データ: s, t, w (全て学習対象)。

    mutation 意味論はここで完結:
      - birth: op が座標と初期 w を運ぶ。moment/計器は FollowerHub 経由で追従
      - death: 論理削除 (SlotPool.release + notify)
      - merge: s,t = w 質量重み平均 / w = 和。mass は自分の w から取る
        (v0.2 の「mass を外から渡す」逆流はこの縦割りで消えた)
      - kick : 座標への in-place 加算
    kernel の per-atom パラメタ (per-atom σ 等) は extras 列として
    add_extra() で追加し、merge 則を登録させる。
    """

    def __init__(self, site: str, d_in: int, d_out: int,
                 capacity: int, rank: int = 0, device=None):
        # self._ids = IdAllocator(rank); self._slots = SlotPool(capacity)
        # self._hub = FollowerHub(); s/t/w は nn.Parameter [capacity, ...]
        ...

    def view(self) -> SynapseView: ...
    def apply(self, op: Op) -> None:
        """SynapseBirth/Death/Merge/Kick を受理。それ以外は TypeError。"""
        ...
    def add_extra(self, name: str, shape: tuple[int, ...],
                  merge: Callable[[Tensor, Tensor], Tensor],
                  learnable: bool = False) -> None: ...


# ══════════════════════════════════════════════════════════════════════
# 3. neuron column — store / view / ops
# ══════════════════════════════════════════════════════════════════════

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
      - gated の c は BN gamma 系の連続振幅 (A6)。death 判定の主対象
      - neuron merge は v0.3 では提供しない (必要になった時に op を足す —
        op 語彙は entity ローカルなので後方互換で追加できる)
    """

    def __init__(self, site: str, coords: Tensor, *, gated: bool = False,
                 learnable_coords: bool = False, capacity: int | None = None,
                 rank: int = 0): ...

    def view(self) -> NeuronView: ...
    def apply(self, op: Op) -> None:
        """NeuronBirth/Death/Kick を受理。"""
        ...


# ══════════════════════════════════════════════════════════════════════
# 4. Compute — 横断点その1: synapse と neuron が出会う場所
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Observation:
    """Compute → Decision。backward の瞬間に site で観測できる生データ。

    x と g_out から dense 勾配 G = g_outᵀ x が非実体化のまま導けるのが
    存在理由 (「まだ無い原子の勾配」は .grad に無い)。hook は detach して
    詰める以上のことをしない。"""

    site: str
    x: Tensor        # 層入力 [B, N_in]
    g_out: Tensor    # 出力勾配 [B, N_out]
    version: int


class Kernel(Protocol):
    """κ の族。global パラメタは自前 nn.Parameter。per-atom パラメタが
    必要なら install(store) で SynapseStore.add_extra を呼んで列を確保する。
    パラメタ不要な kernel は両方とも no-op。"""

    def install(self, store: SynapseStore) -> None: ...
    def global_params(self) -> Iterable[nn.Parameter]: ...
    def __call__(self, query: Tensor, centers: Tensor,
                 extras: dict[str, Tensor]) -> Tensor:
        """κ(query_i − center_k) を評価して [N_query, K] を返す。"""
        ...


class GaussianKernel(nn.Module):
    def __init__(self, sigma: float, *, learnable: bool = True,
                 per_atom: bool = False): ...

class TriangularKernel(nn.Module):
    """引数最小・コンパクトサポートの例。"""


class CSTLinear(nn.Module):
    """横断合成: in/out の NeuronStore × SynapseStore × Kernel。

    forward は View のみから:
      y = gate_out ⊙ (((x ⊙ gate_in) @ K_in) * w) @ K_out.T
      K_in = κ_in(μ_in ⊖ s) [N_in,K], K_out = κ_out(μ_out ⊖ t) [N_out,K]
    (N_in×N_out 非実体化)。View は store.version が動いた時のみ再取得。
    backward hook は Observation を publish する「だけ」— この module は
    Op を発行できない (三角形の辺制約)。"""

    def __init__(self, in_neurons: NeuronStore, out_neurons: NeuronStore,
                 synapses: SynapseStore, kernel_in: Kernel,
                 kernel_out: Kernel | None = None):
        """kernel_out=None なら kernel_in を両側共有。in/out に同じ
        NeuronStore を渡せば座標共有、別 store なら分離。"""
        ...

    def forward(self, x: Tensor) -> Tensor: ...
    def stores(self) -> list[EntityStore]: ...


class CSTConv2d(nn.Module):
    """relative CST (受容野内相対座標を空間共有・channel は k 共有低ランク)。
    v0.3 でも署名のみ確定。"""

    def __init__(self, in_neurons: NeuronStore, out_neurons: NeuronStore,
                 synapses: SynapseStore, kernel: Kernel,
                 rf_size: int, channel_rank: int): ...


# ══════════════════════════════════════════════════════════════════════
# 5. Decision — 計器は entity 縦割り、Policy は横断点その2
# ══════════════════════════════════════════════════════════════════════

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
    def read(self) -> "Reading": ...


class Reading(Protocol):
    """decide() に渡す計器の読み値 (読み取り専用・id ベース)。"""

    def topk(self, k: int, largest: bool = True) -> Tensor: ...  # -> ids
    def value(self, ids: Tensor) -> Tensor: ...


# 同梱計器 (初期 3 policy が要る分だけ):
class GradEMA(Instrument):
    """|∂L/∂w| の EMA (synapse 用)。死亡候補の定番。"""

class GateEMA(Instrument):
    """|c| / |∂L/∂c| の EMA (gated neuron 用)。neuron death の判断材料。"""

class RentCounter(Instrument):
    """rent gate 用の効用/家賃カウンタ (RENT policy の本体)。両 entity 対応。"""

class CandidateProbe(Instrument):
    """birth 候補の off-support 勾配場プローブ (synapse 用; cRigL の本体)。
    候補座標プール (派生データ) を保持し、毎 update で
      score_c += | (g_outᵀ κ_out(t_c)) · (κ_in(s_c)ᵀ x) |
    を dense G 非実体化で一括評価して EMA。read() は座標付き Reading。
    proposal="uniform" | "near_support" | callable。"""


class Policy(Protocol):
    """自由度の唯一の置き場・横断点その2。

    decide は「全 site・全 entity の読み値」を見て混成 op バッチ
    (例 [NeuronDeath(...), SynapseMerge(...)]) を返せる。entity 独立の
    ゲートに分解することを API は強制しない (SC-MRG-1 / SC-LIFE-2)。
    純関数契約: 記録済み Reading でリプレイ可能・乱数は rng 経由のみ (P4)。"""

    def instruments(self) -> dict[str, tuple[Instrument, str]]:
        """{計器名: (instance, bind先 site)}。Engine が bind と配線を行う。"""
        ...

    def schedule(self, step: int) -> list[str]:
        """発火フェーズ名を実行順で (例 ["death", "birth"])。空 = 何もしない。"""
        ...

    def decide(self, phase: str, readings: dict[str, Reading],
               views: dict[str, View], rng: torch.Generator) -> list[Op]: ...


# ══════════════════════════════════════════════════════════════════════
# 6. Engine — 三角形の配線と Executor (ユーザーが触る唯一の入口)
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# 7. 参照 policy — 「各 1 画面」が API の受け入れテスト
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# 8. 合成例
# ══════════════════════════════════════════════════════════════════════

def example_mlp():
    """
    # Storage: entity ごとに store を立てる (縦割りの底)
    n_in  = NeuronStore("in",  input_coords(784))
    n_h   = NeuronStore("h1",  grid_coords(128), gated=True)   # A6 系の c 付き
    n_out = NeuronStore("out", class_coords(10))
    s1    = SynapseStore("l1", d_in=1, d_out=1, capacity=4096)
    s2    = SynapseStore("l2", d_in=1, d_out=1, capacity=4096)

    # Compute: 横断合成 (同じ n_h を両側に渡す = 座標共有)
    model = nn.Sequential(
        CSTLinear(n_in, n_h,  s1, GaussianKernel(0.07)),
        nn.GELU(),
        CSTLinear(n_h,  n_out, s2, GaussianKernel(0.07, per_atom=True)),
    )

    opt = torch.optim.Adam([])        # param 収集と moment 影列は engine が行う
    engine = CSTEngine(model, opt, cRigL(sites=["l1", "l2"]))

    for step, batch in enumerate(loader):
        loss = criterion(model(batch.x), batch.y)
        loss.backward()               # Observation → 計器がここで煮詰まる
        opt.step(); opt.zero_grad()
        engine.step()                 # schedule 発火時のみ三角形が一周する
    """
