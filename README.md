# torchcst — Continuous Sparse Training in PyTorch

DST (Dynamic Sparse Training) が離散マスクをフリップするのに対し、
CST は構造そのものを連続対象 — 符号付き原子測度 — に緩和し、勾配流と
policy 駆動の mutation (birth / death / merge / kick) で進化させる。

> ν = Σ_k w_k δ_(s_k, t_k),   W_ij = Σ_k w_k κ(μ_i^in − s_k) κ(μ_j^out − t_k)

## アーキテクチャ = Engine × Forward × Backward × Policy × Storage

```
             synapse column          neuron column
Storage   │ SynapseStore + ops   │ NeuronStore + ops   │ ← 縦割り (内部自由)
観測状態  │ synapse observers    │ neuron observers    │ ← Policy所有
──────────┼──────────────────────┴─────────────────────┤
Policy    │   prepare / capture / step → MutationPlan       │
Forward   │   CSTLinear = synapse × neuron の合成           │
Backward  │   PyTorch Tensor hook → ModuleGradRecord        │
```

```text
backward: PyTorch hook ─ModuleGradRecord→ Policy-owned observer
step:     Schedule ─UpdateRequest→ Policy ─MutationPlan→ Storage
```

Policyがprepare時にStorageのread portと必要なPyTorch captureをbindする:

- **Forward/Backward は Op を発行できない**(構造を変えられない)
- **Policy は observer stateと構造判断を所有する**
- **Engine は MutationPlan をsiteへrouteするだけ**

公開契約は `torchcst/engine/`, `torchcst/forward/`, `torchcst/backward/`,
`torchcst/policy/`, `torchcst/storage/`に責務ごとに配置されている。

## ストレージ第一原理

- P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
- P2. id は int64・never-reuse・単調増加・上位ビット rank(分散 birth 無調停)。
- P3. slot-indexed付随状態(optimizer moment等)はFollowerとしてmutationに追従。
- P4. mutation はsite-localなop batch単位のトランザクション。
  version単調増加 + opログ。
  分散は op ログの broadcast 同一適用(テンソル同期なし)。
- P5. checkpoint は正準形(compact → id ソート)でバイト決定的。

## 使い方 (設計目標)

```python
import torchcst as tc

n_in  = tc.NeuronStore("in",  input_coords(784))
n_h   = tc.NeuronStore("h1",  grid_coords(128), gated=True)
n_out = tc.NeuronStore("out", class_coords(10))
s1 = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=4096)
s2 = tc.SynapseStore("l2", d_in=1, d_out=1, capacity=4096)

model = nn.Sequential(
    tc.CSTLinear(n_in, n_h,  s1, tc.GaussianKernel(0.07)),
    nn.GELU(),
    tc.CSTLinear(n_h,  n_out, s2, tc.GaussianKernel(0.07, per_atom=True)),
)

# optimizer は Optimizer インスタンスの代わりに factory (params -> Optimizer)
# を渡せる — torch.optim.Adam([]) は空リストで即死するため、param 収集
# (store.parameters() + kernel.global_params()) を終えた engine がこの
# factory を呼んで実体化する。Optimizer インスタンスをそのまま渡した場合は
# 不足分の param を add_param_group で足す。
opt_factory = lambda params: torch.optim.Adam(params, lr=1e-3)
policy = tc.policies.cRigL(
    sites=["l1", "l2"],
    schedule=tc.PeriodicSchedule(every=500, until=50_000),
)
engine = tc.CSTEngine(model, opt_factory, policy)

# ループの順序不変条件: backward → optimizer.step() → engine.step()。
# mutation (engine.step) を backward と optimizer.step の間に入れてはいけない
# — 死んだ原子の stale grad が mutation 後の行 (新生原子) に適用されてしまう。
# zero_grad は「前 step() の後〜次 backward の前」ならどこでもよいが、
# 末尾置きは「.grad が None で始まる」暗黙前提に依存するのでループ先頭に置く。
for step, batch in enumerate(loader):
    engine.optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()               # Policy-owned capture → observer
    engine.optimizer.step()
    engine.step()                 # schedule発火時だけMutationPlanを適用
```

参照 policy は cSET / cRigL (SET・RigL の CST 版)。policy が各 1 画面で
書けることが API の受け入れテスト。

## status

v0 core 動作中: storage (SynapseStore birth/death・NeuronStore 固定標本点)・
CSTLinear matrix-free forward (dense 等価性テスト済)・CSTEngine + cSET
(MassEMA) / cRigL (MassEMA death + CandidateProbe birth) で三角形が一周する
(E2E テストで mutation を跨ぐ訓練を検証)。

Policyが使うbackward情報は、PyTorch hookから得たdetach済み`ModuleGradRecord`である。
SETはcaptureを登録せず、RigLだけがprepare時にgradient captureを登録する。
optimizer後はScheduleが`UpdateRequest`を返し、Policyが`MutationPlan`を作る。

v0 スコープ外 (NotImplementedError 明示): merge/kick・per-atom σ
(add_extra)・neuron mutation・capacity growth・save/load・分散
(world_size>1)・GateEMA/RentCounter (∂L/∂c に pre-gate 値が要る別の capture
設計が要る)。設計の経緯と未決事項は `docs/architecture_workbench.md` を参照。
