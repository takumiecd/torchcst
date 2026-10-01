# Geometry / Pattern / Chart

公開 API は不変な宣言と、Tensor を所有する共通 State に分ける。座標生成、距離、
初期化、射影、更新、輸送は backend の関数が計算する。

```text
geometry/
  spec.py       GeometrySpec / PatternSpec / ChartSpec
  presets.py    宣言を組み立てる純粋な関数
  state.py      GeometryState / PatternState / ChartState、checkpoint
_backends/torch/
  geometry/     execution.py と euclidean / sphere / torus の計算
  patterns/     execution.py と grid / points の計算
  charts/       execution.py と explicit / product / strip の計算
                construction.py は Tensor 点表・乱数生成から State を作る境界
```

旧 Geometry／Chart／SitePattern ABC、種類ごとの Module、旧名の alias、
`_declarations.py`、旧 checkpoint の読み替えは削除した。State に計算メソッドはない。

## 宣言の field

| 宣言 | field | 意味 |
| --- | --- | --- |
| GeometrySpec | `id`, `revision` | 幾何の意味と版。現行 backend は具体型と revision=1 を確認 |
| GeometrySpec | `intrinsic_dim` | 幾何の自由度 |
| GeometrySpec | `embedding_dim` | 観測点の座標幅。具体型から導出 |
| GeometrySpec | `center_parameter_dim` | atom center の保存幅。具体型・表現から導出 |
| GeometrySpec | `metric` | Euclidean または ambient chord。具体型から導出 |
| SphereGeometrySpec | `radius`, `representation`, `chart_margin` | 半径、中心の ambient/intrinsic 表現、除外する反対極の角度 |
| TorusGeometrySpec | `major_radius`, `minor_radius`, `circle_axis` | 大半径、小半径、Chart の結合座標の円周成分 |
| TorusGeometrySpec | `representation`, `chart_margin`, `max_arc_step` | 中心の保存表現、断面の反対極除外、任意の一更新の弧長上限 |
| GridPatternSpec / LinePatternSpec | `shape`, `start`, `spacing`, `revision` | row-major の局所配置。Line は一軸 |
| PointsPatternSpec | `coordinates`, `revision` | 小さな不変点表 |
| ChartSpec | `kind` | `explicit` / `product` / `strip` |
| ChartSpec | `geometry`, `shape`, `axes`, `revision` | 幾何、論理 shape、各論理軸の Pattern、版 |
| ChartSpec | `trainable` | 明示点表だけ学習可能 |
| ChartSpec | `spacing` | 明示格子の補助 metadata。任意点表では None |
| ChartSpec | `tile_shape`, `axis`, `tile_pitch` | Strip の数学的配置。他の kind では None |

`shape` の階数と幾何の次元は別である。例えば `(64, 784)` を Line と
`GridPatternSpec(shape=(28, 28))` で配置すると、行列を三次元空間に置く。
単一 Chart の線形演算では shape を `(out, in)` とする。入出力 Chart の組も維持する。

## 宣言からモデルを作る

```python
import torch
from torchcst import CSTLinear, TriweightSpec, geometry_presets as layout, presets

chart = layout.product(
    shape=(5, 6),
    axes=(layout.line_pattern(5, spacing=0.2),
          layout.grid_pattern((2, 3), spacing=0.3)),
)
layer = CSTLinear(
    chart=chart, atoms=3,
    kernel=presets.radial(presets.fixed_profile(TriweightSpec(), sigma=2.0)),
)
y = layer(torch.randn(4, 6))  # [4, 5]
assert layer.chart.spec.kind == "product"
```

`CSTLinear`／`CSTConv2d` は ChartSpec を受け取って ChartState を所有する。
Tensor の共有が必要なら、既に作った ChartState も明示的に渡せる。
`NormalizedStripLinear` も Strip の ChartSpec または ChartState を受け取る。

`layout.linspace`／`layout.grid` は固定点表の宣言を作る。大きな直積には
`layout.product` を使う。spacing と low/high は排他的であり、暗黙の座標範囲はない。

```python
from torchcst import ChartState, geometry_presets as layout
from torchcst._backends.torch.charts import execution as charts

state = ChartState(layout.strip(
    shape=(5, 3), tile_shape=(2, 3), axis=0, tile_pitch=0.8,
    axes=(layout.line_pattern(5, spacing=0.2),
          layout.line_pattern(3, spacing=0.3)),
))
indices, local = charts.tile_indices(state, 2)  # 端の部分 tile
sites = charts.positions(state, indices)
```

任意の Tensor 点表は `charts/construction.py` の `points(tensor, ...)`、球面上の
乱数点は `sphere_chart(...)` で作る。幾何の演算は
`_backends/torch/geometry/execution.py` の `decode_centers(state, tensor)` 等を使う。

## 状態と数学契約

GeometryState は半径等の固定 scalar buffer、PatternState は軸の原点・間隔か小点表、
ChartState はこれらと任意の明示点表を所有する。Product／Strip は全 Cartesian
点表を保持せず、要求された site のみ生成する。明示点を trainable にした場合は
`state.coordinates` が Parameter になり、通常の autograd で勾配を得る。

Sphere／Torus の metric は ambient chord。intrinsic 表現でも距離計算前に中心を
ambient 座標へ復号する。Strip の tile_shape／axis／tile_pitch は点配置の定義であり、
CUDA の block size や schedule は backend の Recipe に置く。離れた station、端の
部分 tile、円の継ぎ目、円周が一周未満という制約を保持する。

`declaration()` は現在の buffer／点座標の snapshot。scalar 読み戻しや CPU tuple 化は
設定境界のみで行い、forward／backward／CUDA Graph capture 中には行わない。
checkpoint は Tensor と primitive metadata を保存し、`weights_only=True` で復元できる。
shape、点配置の種類、center 表現、学習可否が異なる checkpoint は拒否する。
読み込まれた buffer の半径・間隔・Strip 配置・明示点の幾何制約も検証する。

数学の詳細は [Chart geometry contract](../../../docs/chart-geometry.ja.md)を参照。
