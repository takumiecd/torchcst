# CSTOptimizer

`CSTOptimizer(base_optimizer, model=model, state_adapter=None)` は既存の
PyTorch optimizer を包む。旧 optimizer、config、N/D solver、moment system、
AtomGrad は削除した。旧実験の再現には変更前の Git revision `33504c6` を使う。

## 所有と更新

| 項目 | 所有者 / 処理 |
| --- | --- |
| Parameter | model |
| learning rate、weight decay、optimizer state | base optimizer |
| gradient | backend の通常の autograd |
| 勾配射影・更新規則・vector transport | KernelSpec が定める Parameterization / Geometry。計算本体は backend |
| 更新順序・CST site 発見 | CSTOptimizer |
| scheduler | PyTorch scheduler を wrapper に接続 |

closure があれば現在点で一度評価し、全 group の勾配の有限性を確認する。
atom 勾配を射影し、更新する atom table の旧座標を保存してから base が更新する。
実際の差分 `proposal - old` とその group の現在の `lr` を Kernel に渡し、
更新結果へ vector state を transport して Parameter に書き戻す。

`grad is None` の atom は更新規則も進めない。ゼロ勾配は有効な step として
活動状態の減衰等を進める。`lr=0` では CST の更新規則を止めるが、base optimizer
自身の clock は通常の仕様どおり進む。wrapper 独自の schedule は持たない。

通常の Parameter は base の更新をそのまま使う。各 base が所有する atom だけを
更新するため、model のパラメータを複数 optimizer に分担できる。同じ atom に
複数の CST site が更新規則を割り当てることは拒否する。

wrapper は atom の列を解釈しない。Polar の活動状態の時間単位は group の lr。
optimizer を替えても同じ規則を使うが、学習軌道が同じになるという意味ではない。
raw atom coordinates への weight decay は base の proposal に含まれ、その後
Kernel の規則で射影・調整される。Kernel の正則化とは別なので、必要に応じて
CST group を `weight_decay=0` にする。

## State adapter

| base optimizer | vector として運ぶ state |
| --- | --- |
| SGD | momentum_buffer |
| Adam / AdamW / Adamax | exp_avg |
| RMSprop | momentum_buffer、centered の grad_avg |
| Rprop | prev（前回の勾配） |
| Adagrad / Adadelta | なし |

二乗の蓄積と step clock は元の保存座標系で保持する。これは座標ベースの
optimizer に vector transport を組み合わせた規則であり、内在的な Riemannian
Adam や covariance transport を実装したものではない。

custom optimizer / subclass は state の意味を明示する。

```python
from torchcst import CSTOptimizer, OptimizerStateAdapter

optimizer = CSTOptimizer(
    custom_base,
    model=model,
    state_adapter=OptimizerStateAdapter(vector_keys=("velocity",)),
)
```

指定した parameter-shaped state を 共通 KernelState を使う backend で運ぶ。それ以外の state を
保持することが正しいかは custom optimizer 側の契約である。subclass を名前
だけで既知の state semantics と推定しない。ASGD の平均パラメータや LBFGS の
複数 proposal / closure 再評価は、既定経路では扱わない。

## PyTorch との接続

wrapper は `torch.optim.Optimizer` であり、`param_groups` / `state` は base と
共有する。scheduler、GradScaler、step hook は wrapper を使う。closure は一度
評価し、base へはその結果を返す closure を渡す。`add_param_group()` も base に
委譲する。base を直接 step / load して wrapper と併用しない。

checkpoint は標準の state / param_groups に `cst` manifest を追加する。
optimizer type、vector keys、parameter 名・shape、KernelSpec と CST policy 所有を検証して
復元する。Kernel / Geometry の設定は model checkpoint で復元する。load 後も
base と wrapper の state / groups は共有する。旧 optimizer や raw base の
checkpoint は受け入れない。model を目的の device へ移してから wrapper を作り、
atom Parameter を置き換えた場合は wrapper を作り直す。

## 現在の範囲

Linear / Conv2d の forward backend に依存しない。Euclidean trainable Chart は
処理できる。trainable 非 Euclidean Chart は atom center と異なる更新契約が
必要なので、専用 policy を追加するまで拒否する。

snapshot と更新作業領域は atom parameter 数に比例する。dense weight、
weight-gradient、Hessian は構築しない。有限性検査は device を同期する。
CST policy を含む CUDA Graph capture、Tensor learning rate、differentiable
optimizer step は未対応。`capturable=True` は拒否する。通常の eager CUDA と
foreach / fused base optimizer は別の実行設定である。

raw gradient の非有限値は base 更新前に拒否する。base 更新後の custom policy
failure まで全 optimizer state を rollback するトランザクションではない。
custom policy は shape と有限性の契約を守る必要がある。
