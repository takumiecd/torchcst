# Global Policy × Engine MVP 設計 RFC

> 状態: **議論用ドラフト**
>
> この文書は実装仕様ではない。まず最小の責務境界へ戻り、cSET、cRigL、
> 将来のneuron mutationを同じモデルで説明できるかを、実装前に検証する。
>
> 既存の`architecture_workbench.md`にあるD-001を再検討する。矛盾する箇所は、
> このRFCが採用されるまではどちらも確定事項として扱わない。

## 1. このRFCで決めたいこと

最低限必要な主要コンポーネントを次の4つとする。

1. **Engine**: 学習全体の所有者。model graph、識別、観測routing、clock、
   Policy lifecycle、mutation実行を管理する。
2. **Policy**: 学習全体に1つ存在し、観測を蓄積してモデル全体の構造変更を
   判断する。
3. **Synapse**: synapseのmodel state、identity、View、local mutation意味論を
   所有する。
4. **Neuron**: neuronのmodel state、identity、View、local mutation意味論を
   所有する。

`Observation`、`Operation`、`MutationPlan`、`PolicyState`は主要コンポーネント
ではなく、これら4者の間を流れる値または状態として扱う。

このRFCでは、特に次を決めたい。

- backward hookが取得した事実を、単一のGlobal Policyへどう届けるか。
- Policyがsynapse/neuron/compute relationをどう総合評価するか。
- Policy固有の時間蓄積状態をどこへ置くか。
- Policyが文字列siteやbindingのfor文を意識せず、どうmutation対象を指定するか。
- Observe、Decision、Mutationの境界で何をしてよいか。

## 2. 中心仮説: PolicyはEngineごとに1つ

PolicyはNeuronごと、Synapseごとに取り付けるlocal controllerではない。
一つの学習runを制御する**Global Policy**である。

一つのlinear relationだけでも、観測対象は次の3 Entityにまたがる。

```text
input Neurons ── Synapses ── output Neurons
```

例えばsynapse birthを判断するとき、Policyは次を同時に利用し得る。

- input activation
- outputおよびgrad_output
- input/output neuronの座標やgate
- 現在のsynapse座標、重み、identity
- off-support candidateにおける仮想gradient
- 同じNeuronを共有する別relationからの観測
- モデル全体の構造予算

これらをEntityごとの独立Policyへ先に分割すると、協調が後付けになる。
したがって、最初から次を基本形とする。

```text
Engine 1個 ── Policy 1個
                 ├─ per-synapse PolicyState
                 ├─ per-neuron PolicyState
                 ├─ per-relation PolicyState
                 └─ global PolicyState
```

Policy内部を複数のruleやcomponentへ分割することは許すが、それはGlobal Policyの
実装詳細である。frameworkが最初からEntityごとの独立Policyを要求しない。

## 3. 実行モデル: Observe → Decision → Mutation

### 3.1 Observe

Observeはbackwardで得た事実を処理し、PolicyStateを更新するphaseである。

Observeで行ってよいこと:

- detach済みTensorを使った派生量の計算
- candidate gradient、saliency、utilityの計算
- EMA、counter、histogram、reservoir、candidate poolの更新
- synapse/neuron/relation/globalなPolicyStateの更新
- 高価なraw observationを要約状態へ圧縮すること

Observeで行わないこと:

- Synapse/Neuron stateのmutation
- birth、death、merge、kickの適用
- optimizer step
- Parameterやoptimizer stateのin-place変更
- 通常のautograd gradientの暗黙な書き換え

Observeの制限は「計算してはいけない」ではない。**model stateへ構造的な副作用を
起こしてはいけない**という制限である。

### 3.2 Decision

DecisionはPolicyStateと現在のmodel Viewを使い、どのOperationを実行するかを
決めるphaseである。

Decisionで決める例:

- mutationを発火するか
- 変更数や全体予算をどう配分するか
- deathするidentity
- birthする座標と初期値
- merge/kick対象
- neuron mutationとincident synapse mutationの組み合わせ
- 複数relation間の競合やglobal top-k

Decisionはmodel stateを変更せず、Global `MutationPlan`を返す。可能な範囲で
副作用を持たせず、同じ入力から同じPlanを再生成できる形を目指す。

### 3.3 Mutation

MutationはEngineがPlanをroutingし、Synapse/Neuronがlocal Operationを検証・適用
するphaseである。

Synapse/Neuronが所有するもの:

- Operationの型・shape・identity検証
- slot割り当て
- ID発行
- tensor更新
- version更新
- local mutationの結果

Engineが所有するもの:

- Global Planの全体検証とtransaction coordination
- target routing
- optimizer stateなど外部追従状態の更新
- op log
- clockとschedule

## 4. OperationとObservationの関係

Policyが発行できる各Operationには、その判断根拠が必要である。

```text
Operation capability
  ├─ 必要なraw facts
  ├─ Observeで作るPolicyState
  └─ Decision rule
```

例:

| Operation | 判断根拠の例 | backward Observe |
|---|---|---:|
| SynapseDeath | 現在の`abs(w)` | 不要な場合がある |
| SynapseBirth | off-support candidate gradient | 必要 |
| SynapseMerge | 距離、mass、機能類似度 | Policyによる |
| SynapseKick | 座標gradient、探索utility | Policyによる |
| NeuronDeath | gate、gate gradient、activity utility | Policyによる |
| NeuronBirth | utility、coverage、incident structure | Policyによる |

したがって、「Operationごとに専用Observeを必ず1個作る」という厳密な1対1対応には
しない。

- 一つのObservationが複数Operationの根拠になり得る。
- 一つのOperationが複数種類のObservationを必要とし得る。
- SETのようにbackward Observationを必要としないOperationもある。

代わりに、次を設計上の条件とする。

> 各OperationをPolicyが発行するために、どの事実・PolicyState・Decision ruleが
> 必要かを説明できなければならない。backward由来の事実が必要なら、対応する
> Observe経路を持たなければならない。

Global Policy内部をoperation-oriented componentへ分解する案は有力である。

```text
Global Policy
  ├─ SynapseBirth rule: observe + state + propose
  ├─ SynapseDeath rule: observe? + state? + propose
  ├─ NeuronDeath rule: observe + state + propose
  └─ Global coordinator: proposalの予算配分・競合解消・Plan化
```

ただし、この分解をframeworkの公開抽象にするか、各Policyの実装詳細に留めるかは
未決とする。MVPでは公開抽象を増やさず、cSET/cRigLを通して必要性を確認する。

## 5. Engineのmodel graph registry

Engineは文字列siteの辞書だけでなく、Entityとcompute relationの接続を管理する。

```text
Entity registry
  NeuronRef N0  → Neuron Storage
  NeuronRef N1  → Neuron Storage
  SynapseRef S0 → Synapse Storage

Relation registry
  LinearRef L0 → {
      input_neurons:  N0,
      output_neurons: N1,
      synapses:       S0,
      gradient_provider: ...
  }
```

識別には二つの役割がある。

- `EntityRef`: runtime中にPolicy、Engine、MutationPlanが使うopaqueなtyped handle。
- `SiteKey`: checkpoint、logging、distributed replayで使う安定した外部識別子。

Policyへ文字列`SiteKey`を見せる必要はない。PolicyはObservationやDecisionContextから
受け取った`EntityRef`を、PolicyStateの索引とMutationPlanのtargetに使う。

Global Policyが複数Entityを区別して変更する以上、何らかのidentityは不可避である。
隠すべきなのはidentityそのものではなく、文字列lookup、binding、routingの機械的な
処理である。

## 6. HookとEngineの連携

### 6.1 基本方針

hookは情報取得の手段であり、DecisionやMutationの手段ではない。

```text
PyTorch Tensor hook
  → raw backward facts
  → Engineが事前にbindしたObservationSink
  → Engineがrelationを解決
  → typed Observationを構築
  → Global Policy.observe
```

hook payloadへEngine、Policy、Storage、文字列siteをすべて詰め込まない。

```text
Raw event      = 何が観測されたか
Bound sink     = どのrelationとしてEngineへ届けるか
Engine registry = そのrelationがどのEntityから成るか
```

### 6.2 Global singletonは使わない

「一つのmodel instanceは同時に一つのEngineが所有する」という制約は採用候補である。
しかし、process-globalな`CURRENT_ENGINE`は使わない。

Engine初期化時にrelation専用Sinkを作り、hook closureへ渡す。

```python
sink = engine.make_linear_observation_sink(linear_ref)

def on_backward(grad_output):
    sink(RawLinearBackward(
        input=saved_input,
        output=saved_output,
        grad_output=grad_output.detach(),
        version=saved_version,
    ))
    return None
```

Engineはhook handleを所有し、`close()`で解除する。同じmodelを別Engineへ同時bind
しようとした場合は初期化時に失敗させる。

### 6.3 forwardとbackwardの分離

`CSTLinear.forward()`は数値計算を担当する。学習時にはforwardで作られたoutput
Tensorへbackward hookを登録する必要があるが、PolicyのObserve処理そのものは
backwardまで実行しない。

hook登録をどこへ置くかは次の二案がある。

1. `CSTLinear.forward()`末尾の小さなinstrumentationとして置く。
2. Engineが`module.register_forward_hook()`を外付けし、その中でoutput Tensorへ
   backward hookを登録する。

MVPの優先候補は2である。数値forward本体からinstrumentationを外せるためである。
ただし、module内部の中間値がObservationに必要になった場合の公開方法を確認する。

推論時は`torch.no_grad()`またはEngineのcapture無効化により、output Tensor hookを
登録しない。

## 7. Observation surface

MVPでは万能なObservation hierarchyを作らず、実在するcompute境界に対応した
`LinearObservation`一種類から始める。

```python
@dataclass(frozen=True)
class LinearObservation:
    relation: LinearRef

    input_neurons: EntityFrame[NeuronRef, NeuronView]
    output_neurons: EntityFrame[NeuronRef, NeuronView]
    synapses: EntityFrame[SynapseRef, SynapseView]

    input: Tensor
    output: Tensor
    grad_output: Tensor
    gradients: LinearGradientProvider
```

`EntityFrame`はopaque refとread-only Viewの組であり、可変Storageを公開しない。

Policyはこの一つのObservationからsynapse/neuron/relation/global stateを同時に
更新できる。

```python
def observe(self, state: RigLState, obs: LinearObservation) -> None:
    state.synapses[obs.synapses.ref].observe(...)
    state.neurons[obs.input_neurons.ref].observe_input(...)
    state.neurons[obs.output_neurons.ref].observe_output(...)
    state.global_state.observe(...)
```

Observationには「将来必要そうな情報」を先回りして全部入れない。cRigLと最初の
neuron Policyで不足が実証された情報だけを追加する。

## 8. PolicyStateの所有

PolicyStateには二種類の所有権がある。

- **意味論上の所有者はPolicy**: stateの型、初期化、更新、Decisionでの利用方法を
  Policyが定義する。
- **lifecycle上の所有者はEngine**: 作成されたstate instanceの保持、device移動、
  checkpoint、restore、run終了時の破棄をEngineが管理する。

最小契約:

```python
class Policy(Protocol[StateT]):
    def create_state(
        self,
        model: ModelView,
        rng: torch.Generator,
    ) -> StateT: ...

    def observe(
        self,
        state: StateT,
        observation: LinearObservation,
    ) -> None: ...

    def decide(
        self,
        state: StateT,
        context: DecisionContext,
    ) -> MutationPlan: ...
```

Global stateの例:

```python
@dataclass
class RigLState:
    synapses: EntityTable[SynapseRef, CandidateState]
    neurons: EntityTable[NeuronRef, NeuronUtilityState]
    relations: EntityTable[LinearRef, RelationState]
    global_state: GlobalBudgetState
```

EngineはPolicyStateの中身を解釈しない。

## 9. DecisionContextとMutationPlan

`engine.step()`はGlobal Policyの`decide()`を一度だけ呼ぶ。

```python
def step(self):
    context = DecisionContext(
        clock=self.clock,
        rng=self.rng,
        model=self.model_view(),
    )
    plan = self.policy.decide(self.policy_state, context)
    return self.apply(plan)
```

`DecisionContext.model`はEntityとrelationの現在Viewを提供する。Storage mutation能力は
公開しない。

Policyが返すOperationはlocal payloadとtyped targetを分ける。

```python
@dataclass(frozen=True)
class AddressedOperation:
    target: EntityRef
    operation: SynapseOperation | NeuronOperation
```

```python
@dataclass(frozen=True)
class MutationPlan:
    operations: tuple[AddressedOperation, ...]
```

Engineは`target`をregistryで解決し、対応Storageへlocal Operationを渡す。Policyは
文字列siteを生成しない。

Global Planは、複数Storeのmutationを含み得る。MVPではまず全targetとOperationを
適用前に検証する。neuron deathとincident synapse deathを実装する前に、
prepare/commitを含むtransaction契約を決める。

## 10. 最小lifecycle

### Engine初期化

1. modelからNeuron、Synapse、compute relationを収集する。
2. EntityRef、LinearRef、必要なら安定したSiteKeyを割り当てる。
3. Policyを一度受け取る。
4. `Policy.create_state(ModelView, rng)`を一度呼ぶ。
5. relationごとのObservationSinkを作る。
6. 必要なforward/backward hookを設置する。
7. optimizerとmutation followerを構成する。

### Forward

1. Synapse/Neuron Viewから通常の数値計算をする。
2. captureが必要かつoutputがgradientを要求する場合だけTensor hookを登録する。
3. PolicyのObserve/Decision/Mutationは実行しない。

### Backward

1. Tensor hookがdetach済みraw factsを作る。
2. relation専用ObservationSinkを呼ぶ。
3. Engineがrelationを解決し、forward時に保存したversionと現在versionの一致を
   検証して`LinearObservation`を作る。
4. `Policy.observe(state, observation)`を呼ぶ。
5. PolicyStateだけが更新される。

### Structural step

1. EngineがScheduleを評価する。
2. 発火時だけ`Policy.decide(state, DecisionContext)`を一度呼ぶ。
3. EngineがGlobal MutationPlanを全体検証する。
4. Engineが各Storageへlocal Operationを適用する。
5. Engineがoptimizer state、version、op logを追従させる。
6. PolicyStateにmutation後のreconcileが必要なら、Policy定義の規則で行う。
   EngineはPolicyStateの中身を直接解釈・更新しない。

学習ループ:

```python
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()       # hook → Engine sink → Policy.observe
optimizer.step()
engine.step()         # Policy.decide → MutationPlan → Storage mutation
```

`engine.backward()`とbackward record queueはMVPに含めない。

## 11. MVPに含めるもの

- EngineごとにGlobal Policy 1個
- Neuron StorageとSynapse Storage
- Linear relation registry
- `LinearObservation` 1種類
- EngineがbindするObservationSink
- `Policy.create_state / observe / decide`
- typed EntityRefとlocal Operation
- Global MutationPlan
- Synapse birth/death
- cSET: backward Observeなし
- cRigL: candidate gradient Observeあり
- 1つ以上の共有Neuronを持つ2 relationの観測テスト

## 12. MVPに含めないもの

- Entityごとの独立Policy
- 公開`SynapsePolicy` / `NeuronPolicy` hierarchy
- 公開operation-component framework
- Policy内の文字列sites、ReadPort、binding用for文
- `BackwardContext`、`engine.backward()`、record queue
- 複数subscriber向け汎用pub/sub
- distributed実行
- checkpoint実装
- Conv observation
- merge/kick/neuron mutationの実装
- multi-store prepare/commitの完成実装

## 13. 受け入れ条件

1. PolicyはEngineごとに一つだけ存在する。
2. `Policy.observe()`は一つのLinear relationに関係するinput neurons、output neurons、
   synapsesを同時に読める。
3. `Policy.decide()`はモデル全体を一度に見てGlobal Planを返せる。
4. cSET Policyはbackward hookを必要としない。
5. cRigL Policyはhookから得たObservationでPolicyStateだけを更新する。
6. backward中にNeuron/Synapseのversionが変化しない。
7. Store mutationは`engine.step()`内でしか起きない。
8. Policy実装に文字列site、capture subscription、hook handle、ReadPortが現れない。
9. EngineはPolicyStateの中身を解釈しない。
10. 同じNeuronを共有する複数relationのObservationが、同じGlobal PolicyStateへ
    集約される。
11. 推論時はPolicy Observeが実行されない。
12. MutationPlanのtargetはObservation/DecisionContext由来のtyped EntityRefである。

## 14. 未決事項

### 優先度: 高

1. Policyは必要なObservation種別をどう宣言するか。MVPではPolicy全体の
   `requires_backward`だけで十分か。
2. hook登録をEngine外付けforward hookにするか、Compute実装末尾の
   instrumentationにするか。
3. `LinearObservation`へどこまでraw factsを含め、どこからProviderによるlazy計算に
   するか。
4. EntityRefをruntime-onlyとし、安定SiteKeyとの対応をEngineだけが持つか。
5. PolicyState内のper-entity tableをframework部品として提供するか、Policy実装へ
   任せるか。
6. Decision後、mutation成功時だけPolicyStateをreset/reconcileするための
   `on_commit`が必要か。

### 優先度: 中

7. gradient accumulation時、Observeをmicrobatchごとに行うか、optimizer step単位へ
   集約するか。
8. 同一moduleのreentrant forwardや複数forwardからのObservation順序をどう定義するか。
9. Policy内operation-oriented componentを公開契約にするか。
10. ScheduleはEngine構成かPolicy構成か。更新時刻とアルゴリズム固有の更新量を
    どこまで分けるか。
11. PolicyState checkpointの正準形式をどうするか。

## 15. レビュー時に反証してほしいこと

このRFCをレビューするときは、抽象的な好みではなく次の具体例で反証する。

1. cSETを、Policy内のsite bindingなしで表現できるか。
2. cRigLのoff-support candidate gradientを`LinearObservation`から計算できるか。
3. 同じhidden Neuronを前後2 relationから観測し、utilityを集約できるか。
4. neuron deathと全incident synapse deathを一つのGlobal Decisionで表現できるか。
5. 全relationからglobal top-k birth候補を選べるか。
6. 推論時にEngine/Policy instrumentationを実質的に無効化できるか。
7. PolicyStateをmodel stateと独立にcheckpointできるか。

レビューコメントは、可能なら次の形で残す。

```text
対象節:
判断: Agree / Concern / Reject
具体例:
破綻する理由:
最小の代案:
```

## 16. 次の作業

1. このRFCを人間と複数のmodelでレビューする。
2. 未決事項1〜6へ仮決定を置く。
3. コードを書かずに、cSETとcRigLのPolicy擬似コードをこの契約で作る。
4. 擬似コードから文字列site、binding、routing、hook handleが消えることを確認する。
5. その後にだけ、最小のvertical sliceを実装する。
