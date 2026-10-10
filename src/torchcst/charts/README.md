# Chart の宣言と State

Chart は、Geometry 上で観測する点の配置を宣言する。Geometry、Pattern、Chart は
`torchcst/geometry/`・`torchcst/patterns/`・`torchcst/charts/` の同じ階層に置く。

```text
charts/
  base.py       ChartSpec / ChartState の ABC、共通 metadata と所有の契約
  explicit.py   ExplicitChartSpec / ExplicitChartState
  product.py    ProductChartSpec / ProductChartState
  periodic_grid.py PeriodicGridChartSpec / PeriodicGridChartState
  strip.py      StripChartSpec / StripChartState
  presets.py    純粋な宣言の組み立て
  __init__.py   公開型、compile_chart
_backends/torch/charts/
  execution.py  具体的な State と Spec の型から計算を選ぶ
  explicit.py   明示点の選択・距離・中心の初期化
  product.py    直積の部分展開
  strip.py      Strip の座標・tile・支持域の計算
  lazy.py       Product / Strip が共用する演算
```

## フィールド

| 型 | フィールド | 意味 |
| --- | --- | --- |
| ChartSpec（ABC） | geometry, shape, revision | 空間、論理 shape、意味の版 |
| ExplicitChartSpec | coordinates, trainable, spacing | ambient 座標表、学習の有無、任意の間隔 metadata |
| ProductChartSpec | axes | 各論理軸の Pattern。最後の軸を最速にして平坦化 |
| PeriodicGridChartSpec | grid_shape, output_dims, origin | FlatTorus上の等間隔D次元格子を単一[out,in]へ割当 |
| StripChartSpec | axes, tile_shape, axis, tile_pitch | Pattern、tile の論理 shape、Strip 軸、物理 pitch |

kind は各具体型の固定 ID で、constructor には渡さない。Product / Strip の点配置は固定。
Product には tile 設定や coordinates がなく、Explicit には axes / tile 設定がない。
共通基底には演算メソッドを定義しない。距離・座標展開・微分・更新は backend に置く。

State も具体型に分かれる。ExplicitChartState は座標 Tensor と任意の spacing を所有し、
trainable のときだけ座標を Parameter にする。ProductChartState は小さな PatternState
だけを所有する。StripChartState は PatternState と tile_pitch を所有し、tile_grid /
tile_count は Strip にだけ存在する。共通 ChartState は geometry と形、device/dtype の
参照、declaration / checkpoint の契約だけを持つ。

## 構築と利用

```python
import torch
from torchcst import CSTLinear, TriweightSpec, presets
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
    kernel=presets.radial(presets.fixed_profile(TriweightSpec(), sigma=2.0)),
)
y = layer(torch.randn(4, 6))
assert y.shape == (4, 5)
assert type(layer.chart).__name__ == "ProductChartState"
```

Layer は Spec から対応する State を構築する。既存 State を渡せば共有する。
単一 Chart と入力・出力 Chart の組の両方に対応する。
State を直接構築する場合も具体型を使うか、compile_chart を呼ぶ。

```python
from torchcst import compile_chart, StripChartState
from torchcst._backends.torch.charts import execution

spec = charts.strip(
    shape=(5, 3),
    tile_shape=(2, 3),
    axis=0,
    tile_pitch=0.8,
    axes=(patterns.line(5, spacing=0.2), patterns.line(3, spacing=0.3)),
)
state = compile_chart(spec)
assert isinstance(state, StripChartState)
indices, local = execution.tile_indices(state, 2)
positions = execution.positions(state, indices)
assert positions.shape == (3, 2)
```

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

Strip の tile_pitch は実際の点配置であり、GPU block/window/warp 設定ではない。
部分端 tile、disjoint な物理 interval、Torus の一周未満の配置制約を維持する。
Sphere / Torus の metric と representation は Geometry の宣言に従う。

PeriodicGridChartStateは各軸の原点とGeometryの周期だけを持ち、spacingはperiod/countから
導出する。全site表やPatternStateを持たない。[API・数学・計算方式](../../../docs/periodic-grid.ja.md)。
