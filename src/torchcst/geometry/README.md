# Geometry の宣言と State

GeometrySpec は Euclidean / Sphere / FlatTorus の immutable な数学的宣言。
GeometryState はその metadata を所有し、固定 Tensor buffer を持たない。
点列の Pattern は [patterns/](../patterns/README.md)、観測配置の Chart は
[charts/](../charts/README.md) に分ける。計算は backend の関数で実行する。

| 配置 | 責務 |
| --- | --- |
| `spec.py` | 具体的な GeometrySpec、次元・metric・固定値の検証 |
| `presets.py` | euclidean / sphere / flat_torus の純粋な構築 |
| `state.py` | 固定 metadata、checkpoint、初期化時の placement hints |
| `../_backends/torch/geometry/` | 距離、埋め込み、射影、retraction、vector transport |

| 宣言 | 意味 |
| --- | --- |
| EuclideanGeometrySpec | intrinsic_dim と通常の Euclidean metric |
| FlatTorusGeometrySpec | intrinsic_dim、periods。独立周期座標の最短距離 |
| SphereGeometrySpec | intrinsic_dim、radius、representation、chart_margin |

embedding_dim / center_parameter_dim / metric は具体型から決まる。
Sphere の距離は ambient chord。ambient / intrinsic の中心表現を維持する。
`radius` / `chart_margin` は Python scalar、`periods` は tuple。Module.to() によって
これらの値を丸めない。Torch の計算では引数の Tensor、CUDA の実行では atom Parameter が
実際の device / dtype を定める。必要な device 定数は AlgorithmState の一時キャッシュに置き、
Geometry の学習状態や checkpoint buffer にはしない。

```python
import torch
from torchcst import GeometryState, geometry_presets as geometry
from torchcst._backends.torch.geometry import execution

spec = geometry.sphere(2, radius=2.0, representation="intrinsic")
state = GeometryState(spec, dtype=torch.float64)
assert not tuple(state.buffers())
centers = torch.tensor([[0.1, 0.2]], dtype=torch.float64, requires_grad=True)
sites = execution.decode_centers(state, centers)
assert sites.shape == (1, 3)
execution.squared_distance(state, sites.detach(), centers).sum().backward()
assert torch.isfinite(centers.grad).all()
```

`declaration()` は設定境界で metadata を検証し、forward / backward で host scalar を
読み戻さない。固定値を変更する場合は検証済みの新しい Spec に置き換える。
RegularGrid の周期を変えるときは Chart の spacing と geometry の宣言も整合させる。

checkpoint の extra state は固定値を含む全宣言を保存し、復元先との一致を検証する。
異なる固定値のモデルや旧 Tensor-buffer 形式へ読み替える loader は置かない。
観測点の Parameter は ExplicitChartState、atom の中心は Atoms が所有する。
