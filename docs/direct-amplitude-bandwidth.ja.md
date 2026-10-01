# DirectAmpWidth の座標と更新

DirectAmpWidth は signed amplitude と活動状態を直接保存する。
PolarAmpWidth は角度と二乗半径で表す。どちらも更新規則の計算は backend に置き、
CSTOptimizer は base optimizer の proposal をその規則へ渡す。

```text
Direct: p = (w, q, center...)
w in [-W, W], q in [1, 4], alpha = (q - 1) / 3
Polar: p = (s, t, center...)
q = s^2 + t^2, w = W * s / sqrt(q)
```

Chart の組を使う場合は input / output center が入り、単一 Chart の Direct では
center が一つ入る。各座標の意味と checkpoint は異なる。

## Direct の更新

q の task gradient はゼロに射影する。base optimizer が提案した w の変位を d_w、
現在の learning rate を h とすると、受理された振幅の移動から活動状態を更新する。

```text
w_new = clamp(w + d_w, -W, W)
q_geom = clamp(q + q * ((w_new - w) / W)^2, 1, 4)
q_new = clamp(q_geom - h * radial_regularization * (q_geom - 1), 1, 4)
```

中心座標は Profile / Geometry の更新規則で処理する。Polar の activity gain、
time-energy の除算、dormant expansion は Direct の規則には含まれない。
この処理は AdamW 専用ではなく、SGD 等の proposal にも適用する。

## 使用方法

```python
import torch
from torchcst import CSTOptimizer

base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
optimizer = CSTOptimizer(base, model=model)
optimizer.zero_grad()
loss = model(inputs).square().mean()
loss.backward()
optimizer.step()
```

state の種類と checkpoint は [CSTOptimizer](cst-optimizer.ja.md) を参照。
Direct / Polar の旧 MNIST 比較および旧 optimizer の結果は、Git revision
`33504c6` のこの文書に残っている。新しい wrapper の学習精度を測った結果ではない。
