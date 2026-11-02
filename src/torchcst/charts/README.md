# Chart の宣言と State

Chart は、Geometry 上で観測する点の配置を宣言する。Geometry、Pattern、Chart は
`torchcst/geometry/`・`torchcst/patterns/`・`torchcst/charts/` の同じ階層に置く。

## 型と配置

| 型 | 宣言する配置 |
| --- | --- |
| ExplicitChartSpec | 明示座標、学習の有無、任意の spacing metadata |
| ProductChartSpec | Pattern ごとの軸配置と論理 shape |
| RegularGridChartSpec | 単一階層の grid_shape、origin、spacing |

各 Chart は `ncs` / `num_sites`（`features` と同じサイト総数）を持つ。
`charts=(output_chart, input_chart)` の Weight 形状は `[out.ncs, in.ncs]`。
多次元の RegularGrid も一つの階層であり、`grid_shape=(16,16)` の `ncs` は 256。
Factor 内部では Chart の全次元の距離を使い、サイト全体で正規化する。

`base.py` が共通契約、`explicit.py` / `product.py` / `regular_grid.py` が具体型、
`presets.py` が純粋な構築、`compile_chart` が対応する State を作る。
距離・点展開・支持域・初期化は `_backends/torch/charts/` に置く。

ExplicitChartState は座標 Tensor を持ち、trainable のときだけ Parameter にする。
ProductChartState は PatternState を所有する。RegularGrid は全サイト表を持たず、
インデックスから座標を生成する。origin / spacing と幾何の固定値は Python metadata として
保持し、Tensor buffer を持たない。Module.to() は初期化用 device / dtype hints を移すが、
固定値は丸めない。実行時の距離は atom の中心 Tensor の placement に従う。
checkpoint は固定 metadata の一致を検証し、異なる配置へ読み替えない。

## 構築と利用

```python
import torch
from torchcst import CSTLinear, TriweightSpec, kernels
from torchcst import (
    chart_presets as charts,
    geometry_presets as geometry,
    pattern_presets as patterns,
)

chart = charts.product(
    shape=(5, 6),
    axes=(patterns.line(5, spacing=0.2), patterns.grid((2, 3), spacing=0.3)),
    geometry=geometry.euclidean(3),
)
layer = CSTLinear(
    chart=chart,
    atoms=3,
    kernel=kernels.radial(kernels.point(TriweightSpec(), sigma=2.0)),
)
y = layer(torch.randn(4, 6))
assert y.shape == (4, 5)
assert type(layer.chart).__name__ == "ProductChartState"
```

Layer は Spec から対応する State を構築する。既存 State を渡せば共有する。
単一 Chart と入力・出力 Chart の組の両方に対応する。
State を直接構築する場合も具体型を使うか、compile_chart を呼ぶ。

Explicit は PointPattern で包まず座標表を直接宣言する。shape は座標表の長さと一致する
論理 shape で、明示した行列配置にも対応する。presets.points / grid / linspace は
一次元の点列を作る。大きい直積には presets.product を使い、全点表を作らない。

```python
from torchcst import ExplicitChartSpec, ExplicitChartState

explicit = ExplicitChartSpec(
    geometry=geometry.euclidean(2),
    shape=(2, 3),
    trainable=True,
    coordinates=tuple((float(i), float(j)) for i in range(2) for j in range(3)),
)
state = compile_chart(explicit)
assert isinstance(state, ExplicitChartState)
assert isinstance(state.coordinates, torch.nn.Parameter)
```

## 境界と検証

declaration は現在の Tensor の設定 snapshot。forward / backward / Graph capture では
呼ばない。演算は live な Tensor を使う。checkpoint は具体型、shape、版、trainable、
その型の構造を検証し、weights_only でロードできる。別種類の Chart や旧 tagged ChartSpec /
ChartState の形式を読み替える loader はない。ChartSpec / ChartState の直接構築も廃止した。

closed な compile_chart と backend は、具体 Spec / State の組と revision を照合する。
未登録の subclass が kind を引き継いでも組み込みとして実行しない。拡張は宣言型と
対応 backend を明示的に追加する。

Sphere / FlatTorus の metric と representation は Geometry の宣言に従う。

`charts.regular_grid((16,16), spacing=0.1)` は一つの 2D Chart、shape `(256,)` を宣言する。
最後の座標軸を最速にして平坦化する。scalar の spacing / origin は各座標へ broadcast し、
宣言型には canonical な tuple を保存する。全サイト表・全サイトの index 表は保存しない。
`geometry="flat_torus"` は各周期を `grid_shape * spacing` から導出する。
