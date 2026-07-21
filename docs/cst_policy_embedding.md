# cst研究成果のPolicy埋め込み exercise

> **履歴文書:** pre-v4 RFCに対する実現可能性検討であり、型名や擬似コードは
> 現行APIではない。現在の位置づけは [`docs/README.md`](README.md) を参照すること。
>
> 状態: 議論用ドラフト。実装仕様ではない。
>
> [`policy_engine_mvp_rfc.md`](policy_engine_mvp_rfc.md) の契約
> (`Policy.observe / decide`、`EntityRef`、`LinearObservation`) の上で、
> `../cst` の構造制御研究 (`cst.structure`) をPolicyとして書けるかを
> 擬似コードで検証する。RFC 15節の反証課題に、この文書の結果を追加する。
>
> **持ち込むのは数式と意味論であって、コードではない。** `cst`は実験repoで
> あり、実装は数式の成立を確認するための装置に過ぎない。torchcstでは全部品を
> torchcstの語彙で新規に設計・実装し、`cst`の実装は数値同値テストのoracle
> としてだけ使う。

## 1. 前提: 2つのrepoは相補的である

`cst`と`torchcst`は競合する実装ではなく、同じ問題の別の半分を持っている。

| | `cst` (研究) | `torchcst` (基盤) |
|---|---|---|
| 候補の提案・screening・評価 | **ある** (`StructuralOperation`, `StructuralController`) | ない |
| 候補の共通表現 | **ある** (`CandidateBatch` / `TrialBatch`, M軸規約) | ない |
| 構造コストと利益の共通単位 | **ある** (`LinearCost`, profit = gain − cost) | ない |
| shadow評価 (候補worldの実loss) | **ある** (`shadow_weight_losses` 等) | ない |
| live mutation | **ない** (observe-onlyで停止、live applyは未開放) | **ある** (Store / Op / SlotPool) |
| optimizer state追従 | 実験的 (`live_neuron`のAdamW移行) | **ある** (`Follower`) |
| entity identityとversion | ない (文字列target + 密行列) | **ある** (ids / version / View) |
| 訓練backwardからの観測 | 別forwardでprobe (`probe_objective`) | **ある** (hook capture) |

つまり統合の切れ目は自然に決まる:
**`TrialBatch`(判断)までがcst由来、`Op`(変異)からがtorchcst。**
`cst`が「live applyをまだ有効化していない」のは、まさにtorchcstが所有する
mutation基盤が欠けているからであり、この統合がそのギャップを埋める。

## 2. 部品の対応表

| cst | torchcst側の対応 | 判定 |
|---|---|---|
| `ProbeBatch` (input / output / grad_output / loss) | `LinearObservation` (input / output / grad_output) | ほぼ同型。RFCの観測面が研究要件と一致している証拠 |
| `probe_objective()` (専用forward + `autograd.grad`) | 訓練backwardへの相乗り (hook capture) | torchcst方式が上位互換。decide時のfresh probeはTrial機構(後述)で可能 |
| `probe_weight_gradient()` / `grid_birth_scores()` | `LinearGradientProvider.weight_gradients()` | 同じ数学。cstはneuron格子(`mu_out × mu_in`)上、torchcstは任意連続座標。格子はViewの`mu`から作れるので包含される |
| `CandidateBatch` / `TrialBatch` | なし | **M軸規約という設計思想を採り、型は再設計** (後述: payload辞書は持ち込まない) |
| `LinearCost` (delta = atoms/synapses/neurons) | なし | **意味論(共通profit単位)を採って再実装**。RFCのglobal予算の実体になる |
| `AmplitudeProblem` (振幅refit) | なし | **数式を仕様化して再実装** (純tensor代数。数値oracleにcst実装を使う) |
| `StructuralOperation` (propose / evaluate) | なし | propose/evaluateという2段の意味論だけ採る。**公開Engine契約にはしない** (RFC未決9の答え) |
| `StructuralController._select_globally()` | なし | screen意味論(profit単位の横断top-k)を純関数として再実装。Global Policyの`decide()`本体になる |
| `ProgressiveNeuronSchedule` | なし | 判断規則(profit gate、rejection打ち切り)を仕様化して再実装 |
| `shadow_weight_losses()` / `activation_shadow_losses()` | なし | **hook配管は移植しない**。Engine提供のTrial機構として再設計 (後述のギャップB) |
| `_resolve_layer(model, "name")` 文字列target | `EntityRef` / relation registry | cst側の文字列解決は捨てて、RFCのregistryに置換 |
| `live_neuron.apply_neuron_birth()` + AdamW state移行 | `Store.apply()` + `Follower` | cst側は捨てる。torchcstが正式な所有者 |
| `NeuronGate` (論理幅の開閉) | `NeuronView.gate` + `NeuronBirth/Death` | 概念一致。物理幅固定 + 論理slotはSlotPoolの思想と同じ |

結論: **研究の判断アルゴリズムは全量、torchcstのPolicy層で表現できる。**
ただし持ち込む単位は「数式・判断規則・tensor batch規約」であり、cstの型や
関数をimport/コピーすることではない。modelへの触り方(文字列target、hook
配管、直接apply)は、RFCが「Engineが所有する」と決めた部分に置き換わる。

### cstの実装をそのまま持ち込んではいけない理由 (意図の明文化)

`cst`は「実装が汚くても数式が成り立てばいい」方針の実験repoである。実際、
その汎用コンテナはworkbench 11節でtorchcstが廃止した反パターンを含む。

- `CandidateBatch.payload: Mapping[str, Tensor]` — 文字列キーの辞書へ型を
  隠す。`Reading`/`cast` registryや`DecisionContext.views: dict[str, View]`
  と同種であり、bind後のnarrowingを復活させてしまう。
- `operation: str` / `target: str` — 文字列識別。RFCが`EntityRef`で消した
  ものの再導入になる。
- `delta: Tensor (M, 3)` + `axes: tuple[str, ...]` — 位置合わせを実行時検証
  に頼る資源vector。

torchcstでは候補をfamilyごとのtyped batchとして再設計する。例えば:

```python
@dataclass(frozen=True)
class SynapseBirthCandidates:   # M軸規約は共通、fieldは family固有・typed
    relation: LinearRef
    s: Tensor            # (M, d_in)
    t: Tensor            # (M, d_out)
    w: Tensor            # (M,)  refit済み初期振幅
    predicted_gain: Tensor  # (M,)
    delta: ResourceDelta    # typed資源変化 (atoms/synapses/neurons)
    valid: Tensor           # (M,) bool
```

M軸規約(先頭次元が常に候補軸、`M=1`も同経路)とprofit単位の共通化は
cstの発明として残し、それを**typedに言い直す**のがtorchcst側の仕事である。

## 3. RFC契約に足りないもの: 2つのギャップ

擬似コードを書く前に、RFCの現契約では書けない箇所を明示する。
これがこのexerciseの最重要の発見である。

### ギャップA: `decide()`がdata batchとobjectiveを必要とする

`StructuralController.step(model, batch, objective)`は、候補評価のために
**新しいdata batchでforwardを回す**。一方RFCのlifecycleは

```python
optimizer.step()
engine.step()        # Policy.decide → Plan → mutation
```

で、`engine.step()`にbatchが渡らない。cSET/cRigLはこれで足りるが、
profit駆動の研究Policyは足りない。

提案: `engine.step()`を2形態にする。

```python
engine.step()                          # Trial不要なPolicy (cSET / cRigL)
engine.step(batch=batch, objective=objective)  # Trial要求Policyはこちら
```

Policyは`requires_backward: bool`と対にして`requires_trial: bool`を宣言し、
`requires_trial=True`のPolicyへbatchなしで`step()`した場合はEngineが即座に
失敗させる。

### ギャップB: 候補world評価 (shadow) はEngine capabilityにする

cstのshadow評価は「forward hookで対象層の出力を候補worldの計算に差し替え、
下流を候補軸Mごと流す」方式で、契約は実験で検証済みである。この**方式**は
持ち込むが、**配管**はPolicyに書かせない。DecisionContextに`trial`を足す:

```python
class TrialSession(Protocol):
    def baseline(self) -> Tensor:
        """現在構造でのobjective値。loss_beforeに使う。"""

    def shadow(
        self,
        relation: LinearRef,
        candidates: SynapseWorldBatch | GateWorldBatch,  # M軸の候補world
    ) -> Tensor:
        """各候補worldの実objective値 (M,)。model stateは変更しない。"""
```

実装はEngineがrelation registryから対象moduleを引き、cstと同じ
output差し替えhookで実現する。Policyから見えるのは
「候補Viewの束 → 実lossの束」という純関数だけになる。

重要な設計上の帰結: **候補world = 差し替えられたView** と定義できる。
`CSTLinear`は毎forwardでStoreのViewから計算するので、shadowとは
「そのrelationだけ別のView(候補synapse座標・重み、または別のgate)で
computeすること」に他ならない。cstでは密行列Wの差し替えだったものが、
torchcstではViewの差し替えとして、Storage境界と同じ語彙で表現できる。

このTrialは3相モデルを壊さない。Observe(観測でstate更新)、
Decision(Plan生成)、Mutation(apply)のうち、TrialはDecisionの内部で使う
**read-onlyのforward実行capability**であり、model stateへの副作用はない。
RFC 3.2の「Decisionはmodel stateを変更しない」はそのまま成立する。

## 4. 擬似コード: 研究成果を全部入りにしたPolicy

RFC契約 + ギャップA/Bの拡張で、`cst.structure`の全パイプライン
(propose → global screen → shadow evaluate → profit-gated select)を書く。

```python
class CSTStructural(Policy):
    """grid birth / prune / merge / neuron gate birth-death を
    共通profit単位でglobal screenする研究Policy。"""

    requires_backward = True   # 訓練backwardから勾配signalを蓄積する
    requires_trial = True      # decide時にshadow評価を行う

    def __init__(self, cost: LinearCost, trial_width: int = 16):
        self.cost = cost
        self.trial_width = trial_width
        self.signals: EntityTable[LinearRef, GradSignal] = EntityTable()
        self.neuron_schedules: EntityTable[NeuronRef, ProgressiveNeuronSchedule] = ...
        self.rng = ...

    # ---- Observe: 訓練backwardへの相乗り (cstのprobe_objectiveの代替) ----

    def observe(self, obs: LinearObservation) -> None:
        # cst: probe_weight_gradient(probe) を専用forwardで計算していた。
        # torchcst: 通常のloss.backward()から同じ事実がタダで届く。
        self.signals[obs.relation].update(
            obs.input, obs.grad_output, obs.gradients
        )   # EMAまたは直近K件。中身はGradEMA/CandidateProbeと同系。

    # ---- Decision: StructuralController.step()の移植 ----

    def decide(self, ctx: DecisionContext) -> MutationPlan:
        # 1. propose: relation × operation familyの候補をM軸tensorで作る
        candidates: dict[OpKey, CandidateBatch] = {}
        for ref, rel in ctx.model.linears():
            sig = self.signals[ref]
            candidates[(ref, "grid_birth")] = grid_birth_candidates(rel, sig)
            candidates[(ref, "prune")]      = prune_candidates(rel)        # atomic_prune_candidates
            candidates[(ref, "merge")]      = merge_candidates(rel)        # atomic_merge_candidates
        for nref, neu in ctx.model.neurons():
            candidates[(nref, "gate_birth")] = gate_birth_candidates(neu, ctx.model)
            candidates[(nref, "gate_death")] = gate_death_candidates(neu, ctx.model)

        # 2. global screen: 全relation × 全familyを共通profit単位でtop-k
        #    (StructuralController._select_globallyの純関数移植。
        #     cstはfamily横断のみだったが、relation横断へ自然に拡張される)
        costs = {key: self.cost(c.delta) for key, c in candidates.items()}
        selected = select_globally(candidates, costs, self.trial_width)

        # 3. shadow evaluate: 選抜候補だけ実lossで検証 (ギャップB)
        base = ctx.trial.baseline()
        trials: dict[OpKey, TrialBatch] = {}
        for key, index in selected.items():
            picked = candidates[key].index_select(index)
            worlds = candidate_worlds(picked)          # 候補 → View差し替えの束
            loss_after = ctx.trial.shadow(key.relation, worlds)
            trials[key] = TrialBatch.from_losses(base, loss_after, costs[key][index])

        # 4. profit-gated select (ProgressiveNeuronSchedule.selectの移植)
        chosen = self._gate_by_profit(ctx.clock, trials, candidates)

        # 5. Opへ変換。ここで初めてtorchcstの語彙になる
        ops: list[AddressedOperation] = []
        for key, picked in chosen.items():
            ops += to_operations(picked)
            # grid_birth  → SynapseBirth(s, t, refit済みw)
            # prune       → SynapseDeath(ids)
            # merge       → SynapseMerge(src_ids, dst_id, mass)
            # gate_birth  → NeuronBirth(...) + 付随SynapseBirth (同一Plan内)
            # gate_death  → NeuronDeath(...) + incident SynapseDeath (同一Plan内)
        return MutationPlan(tuple(ops))
```

補助部品の出自 (いずれもcstの数式を仕様として新規実装する):

- `grid_birth_candidates`: `top_grid_birth_proposals` + `AmplitudeProblem`
  refitの数式。signalはObserve蓄積から取るので専用forwardが消える。
- `prune_candidates` / `merge_candidates`: `atomic_prune_candidates` /
  `atomic_merge_candidates`の縮約数式。密行列参照はViewの`(s, t, w)`参照へ。
- `select_globally`: `_select_globally`のscreen意味論を純関数として実装。
- `_gate_by_profit`: `ProgressiveNeuronSchedule`の判断規則を
  per-neuron-site stateとして実装し、Policyの`state_dict()`に載せる。

### 確認できること

- 文字列site、binding for文、hook handle、ReadPortが**一切現れない**
  (RFC受け入れ条件8)。
- neuron death + incident synapse deathが1つの`MutationPlan`に載る
  (RFC反証課題4)。multi-store atomicityの要求が実例で確定する。
- 全relation横断のglobal top-k birth (RFC反証課題5) は、cstで実証済みの
  `select_globally`がそのまま答えになる。
- cSET/cRigLはこのPolicyの退化形になる: cSETは
  `requires_backward=False, requires_trial=False`で1と5だけ、cRigLは
  Observeあり・Trialなしで2の代わりにEMA top-k。つまり**Policy契約は
  1本で、研究Policyと軽量Policyは宣言フラグだけで住み分ける**。

## 5. 共通化 / 分離の判定

### 数式・意味論として持ち込み、torchcst語彙で新規実装する

`torchcst.policy.candidates` (仮) に、次の**仕様**を実装する。
torch-onlyでStorage/Engineに依存しない層とする。

- M軸候補batch規約 — familyごとのtyped batch (2節末の再設計方針)。
- 共通profit単位とtyped `ResourceDelta` — `LinearCost`の意味論。
- 振幅refit — `AmplitudeProblem`の数式 (residual/Gram/joint solve)。
- profit横断screen — `select_globally`の意味論。
- profit gate state machine — `ProgressiveNeuronSchedule`の判断規則。

いずれもcst側でmodel/nn.Module非依存に切り出せることは確認済みなので、
**数値同値テストのoracleとしてcst実装を参照する**ことができる。
つまり「同じ入力tensorに対しcstの関数と同じ値を返す」ことをテストで固定
しつつ、型・命名・境界はtorchcstで新規に設計する。判断アルゴリズムの
資産はこの層に集まる。

### 書き換えて吸収する (境界を跨いでいた部分)

- 文字列target解決 (`_resolve_layer`) → relation registry + `EntityRef`。
- probe専用forward (`probe_objective`) → 訓練backward相乗りのObserve。
- shadow hook配管 → Engine提供`TrialSession`。
- 密行列前提の候補評価 → View語彙 (`s`, `t`, `w`, `gate`, `mu`) へ。

### 捨てる (torchcstが正式な所有者)

- `live_neuron.apply_neuron_birth` と optimizer state手術
  → `Store.apply` + `Follower`。
- `NeuronGate`の物理実装 → `NeuronStore` + gate付き`NeuronView`。

### torchcst側に足す必要があるもの

1. `engine.step(batch, objective)`形態と`requires_trial`宣言 (ギャップA)。
2. `TrialSession` (View差し替えshadow実行) (ギャップB)。
3. Observeの蓄積部品を`EntityTable`ベースへ (RFC未決5は「framework提供」で確定させたい)。
4. `SynapseMerge` / `NeuronBirth` / `NeuronDeath`のStore実装を
   Phase 3から前倒しで語彙だけ確定 (Op型は既にある)。

## 6. RFCへの反映提案

1. RFC 15節の反証課題に追加:
   **「`StructuralController`のpropose → global screen → shadow evaluate →
   profit selectを`decide()`1回で表現できるか」** — 本文書4節が肯定の証拠。
2. RFC未決事項1の答え: `requires_backward`に加え`requires_trial`の2 boolで
   MVPは足りる。
3. RFC未決事項9の答え: operation-oriented component (`propose/evaluate`) は
   **公開契約にせず**、`policy.candidates`層のライブラリ規約に留める。
   `CandidateBatch`のM軸規約だけが実質的な共通契約になる。
4. RFC 12節「MVPに含めないもの」の確認: TrialSessionはMVP必須ではない
   (cSET/cRigLは使わない)。ただし`DecisionContext`にfieldを増やせる形
   (dataclass拡張) だけ確保しておく。
5. multi-store atomicity (workbench 8節): gate_death → NeuronDeath +
   incident SynapseDeathが実要件として確定したので、「全batch検証 → 全apply」
   の2段を最低ラインとし、prepare/commitはその失敗例が出てから。

## 7. 進め方

RFCのMVP (cSET/cRigL) を先に完成させ、その直後に本文書のPolicyを
**擬似コードのままRFC契約に対して型チェック的に照合する** (コード化は
`candidates`層の移植テストから)。順序:

1. RFC MVP: registry / ref / `LinearObservation` / cSET / cRigL。
2. `torchcst.policy.candidates`: 5節の仕様を新規実装。cst実装を数値oracle
   とする同値テストを先に書き、数式の同一性を固定してから型を設計する。
3. ギャップA/B: `requires_trial` + `TrialSession`。
4. 本Policyのvertical slice: まずgrid_birth + pruneの2 familyだけで
   E2E (shadow評価つきbirth/death) を通す。
5. merge / gate系familyを追加し、multi-store Planの検証を確定する。
