# torchcst — Continuous Sparse Training in PyTorch

DST (Dynamic Sparse Training) が離散マスクをフリップするのに対し、
CST は構造そのものを連続対象 — 符号付き原子測度 — に緩和し、勾配流と
policy 駆動の mutation (birth / death / merge / kick) で進化させる。

> ν = Σ_k w_k δ_(s_k, t_k),   W_ij = Σ_k w_k κ(μ_i^in − s_k) κ(μ_j^out − t_k)

## アーキテクチャ = 三角形 × entity 縦割り

```
             synapse column          neuron column
Storage   │ SynapseStore + ops   │ NeuronStore + ops   │ ← 縦割り (内部自由)
計器      │ synapse Instruments  │ neuron Instruments  │ ← 縦割り
──────────┼──────────────────────┴─────────────────────┤
Policy    │   横断: 全計器の読み値 → 協調 op バッチ       │
Compute   │   横断: CSTLinear = synapse × neuron の合成   │
```

```
   Storage ──View──→ Compute ──Observation──→ Decision ──Op──→ Storage
```

辺の型は 3 つだけ。頂点間はこれ以外で会話しない:

- **Compute は Op を発行できない**(構造を変えられない)
- **Decision は View の読みと Op の発行のみ**(重みテンソルに触れない)
- **Engine は Op の中身を見ない**(site-local batchへルーティングして
  applyするだけ)

契約面はすべて `torchcst/contracts.py` に集約されている。

## ストレージ第一原理

- P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
- P2. id は int64・never-reuse・単調増加・上位ビット rank(分散 birth 無調停)。
- P3. per-atom 付随状態(moment/計器)は Follower として mutation に自動追従。
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
engine = tc.CSTEngine(model, opt_factory, tc.policies.cRigL(sites=["l1", "l2"]))

# ループの順序不変条件: backward → optimizer.step() → engine.step()。
# mutation (engine.step) を backward と optimizer.step の間に入れてはいけない
# — 死んだ原子の stale grad が mutation 後の行 (新生原子) に適用されてしまう。
# zero_grad は「前 step() の後〜次 backward の前」ならどこでもよいが、
# 末尾置きは「.grad が None で始まる」暗黙前提に依存するのでループ先頭に置く。
for step, batch in enumerate(loader):
    engine.optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()               # Observation → 計器がここで煮詰まる
    engine.optimizer.step()
    engine.step()                 # schedule 発火時のみ三角形が一周する
```

参照 policy は cSET / cRigL (SET・RigL の CST 版)。policy が各 1 画面で
書けることが API の受け入れテスト。

## status

v0 core 動作中: storage (SynapseStore birth/death・NeuronStore 固定標本点)・
CSTLinear matrix-free forward (dense 等価性テスト済)・CSTEngine + cSET
(MassEMA) / cRigL (MassEMA death + CandidateProbe birth) で三角形が一周する
(E2E テストで mutation を跨ぐ訓練を検証)。

「計器から Kernel へのアクセス経路が無い」という初期の設計未解決点は
**KernelPort** (bind 時に Engine が計器へ渡す読み取り専用の評価能力。
Observation には kernel を同梱しない — per-step の辺は生データのまま不変に
保つ) で解決し、GradEMA / CandidateProbe を実装済み (`contracts.py` の
KernelPort docstring・`decision/instruments.py` 参照)。

v0 スコープ外 (NotImplementedError 明示): merge/kick・per-atom σ
(add_extra)・neuron mutation・capacity growth・save/load・分散
(world_size>1)・GateEMA/RentCounter (∂L/∂c に pre-gate 値が要る別の port
設計が要る、KernelPort では解決しない)。設計の経緯は `docs/api_draft_v0_3.py`
を参照。
