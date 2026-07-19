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
- **Engine は Op の中身を見ない**(site でルーティングして apply するだけ)

契約面はすべて `torchcst/contracts.py` に集約されている。

## ストレージ第一原理

- P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
- P2. id は int64・never-reuse・単調増加・上位ビット rank(分散 birth 無調停)。
- P3. per-atom 付随状態(moment/計器)は Follower として mutation に自動追従。
- P4. mutation は store へのトランザクション。version 単調増加 + op ログ。
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

engine = tc.CSTEngine(model, opt, tc.policies.cRigL(sites=["l1", "l2"]))

for step, batch in enumerate(loader):
    loss = criterion(model(batch.x), batch.y)
    loss.backward()      # Observation → 計器がここで煮詰まる
    opt.step(); opt.zero_grad()
    engine.step()        # schedule 発火時のみ三角形が一周する
```

参照 policy は cSET / cRigL (SET・RigL の CST 版)。policy が各 1 画面で
書けることが API の受け入れテスト。

## status

設計骨格の段階 (v0.3 契約確定・実装は stub)。設計の経緯は
`docs/api_draft_v0_3.py` を参照。
