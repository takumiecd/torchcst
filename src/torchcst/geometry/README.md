# Geometry の宣言と State

`geometry/` は空間の意味と固定 scalar 状態を持つ。
点列の Pattern は [patterns/](../patterns/README.md)、観測配置の Chart は
[charts/](../charts/README.md) に分け、同じ階層に置く。

```text
geometry/
  spec.py       GeometrySpec、EuclideanGeometrySpec、SphereGeometrySpec、TorusGeometrySpec
  presets.py    euclidean / sphere / torus の純粋な構築
  state.py      GeometryState の scalar buffer と checkpoint
_backends/torch/geometry/
  execution.py  宣言型・版に対応する計算を選ぶ
  euclidean.py  Euclidean の距離・更新
  sphere.py     Sphere の埋め込み・距離・射影・更新・輸送
  torus.py      Torus の埋め込み・距離・射影・更新・輸送
```

GeometrySpec は immutable な宣言。GeometryState は半径などの Tensor を所有する。
Geometry / State に距離や演算メソッドを置かない。revision と具体型が対応する
backend の関数を使う。

| 宣言 | 意味 |
| --- | --- |
| EuclideanGeometrySpec | intrinsic_dim と通常の Euclidean metric |
| SphereGeometrySpec | intrinsic_dim、radius、representation、chart_margin |
| TorusGeometrySpec | intrinsic_dim、major_radius、minor_radius、circle_axis、representation、chart_margin、max_arc_step |

embedding_dim / center_parameter_dim / metric は具体型から決まる metadata。
Sphere / Torus の距離は ambient chord であり、geodesic ではない。
ambient / intrinsic の center 表現を維持する。

```python
import torch
from torchcst import GeometryState, geometry_presets as geometry
from torchcst._backends.torch.geometry import execution

spec = geometry.sphere(2, radius=2.0, representation="intrinsic")
state = GeometryState(spec, dtype=torch.float64)
centers = torch.tensor([[0.1, 0.2]], dtype=torch.float64, requires_grad=True)
sites = execution.decode_centers(state, centers)
assert sites.shape == (1, 3)
execution.squared_distance(state, sites.detach(), centers).sum().backward()
assert torch.isfinite(centers.grad).all()
```

state.declaration() は設定境界で現在の scalar を snapshot する。
通常の forward、backward、Graph capture で scalar を host に戻さない。
checkpoint は構造と現在の scalar を検証する。学習可能な観測点の所有は
ExplicitChartState、atom の中心の所有は Atoms にある。
