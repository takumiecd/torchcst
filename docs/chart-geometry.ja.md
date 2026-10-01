# Chart / Geometry の数学契約

GeometrySpec は空間・次元・距離・center の保存表現を宣言する。ChartSpec はその
空間の観測点配置を宣言する。GeometryState／ChartState／PatternState は Tensor を
所有し、計算は backend の関数が実行する。field と構築 API は
[geometry のガイド](../src/torchcst/geometry/README.md)を参照。

## 次元と配置

`intrinsic_dim` は幾何の自由度、`embedding_dim` は観測点の座標幅、
`center_parameter_dim` は atom center の保存幅である。
EuclideanGeometrySpec では三つとも同じ。SphereGeometrySpec は `S^d` を
`R^(d+1)` に埋め込み、ambient center は d+1 個、intrinsic center は d 個を保存する。

Chart の `shape` は論理 Tensor の形であり、幾何の次元とは別である。
例えば `shape=(64, 784)` の二軸に Line と `GridPatternSpec(shape=(28, 28))` を
指定すると、行列を三次元空間に配置する。各 pattern 内も論理軸も row-major。
最後の軸を最速で平坦化する。Chart に input／output という特別な軸名はない。

単一 Chart の線形演算では shape を `(out, in)` とし、Kernel は radial に合成する。
入出力 Chart の組ではそれぞれの点集合を separable に合成する。両形式を保持し、
OperatorSpec の SingleChartSpec／ChartPairSpec で区別する。

```python
from torchcst import geometry_presets as layout

axes = (layout.line_pattern(64, spacing=0.1),
        layout.grid_pattern((28, 28), spacing=2 / 27))
product = layout.product(shape=(64, 784), axes=axes)
strip = layout.strip(shape=(64, 784), axes=axes,
                     tile_shape=(8, 784), axis=0, tile_pitch=4.1)
```

Product は各軸の直積。Strip は選択した Line 軸に station の位置を加える。
延長軸だけを tile 分割し、新たな幾何座標軸を追加しない。tile_shape と tile_pitch は
数学的な点配置であり、CUDA の block size／launch の設定ではない。
両形式とも必要な site 番号の座標だけを backend で生成する。全点表を保持しない。
端の部分 tile も `charts.tile_indices(state, station)` で扱う。

明示点の Chart は一つの PointsPatternSpec から作る。trainable=True の場合、
ChartState.coordinates が Parameter になり、通常の autograd で勾配を得る。
Product／Strip の軸配置は現在固定である。

## Sphere の距離と保存表現

距離は ambient chord `||site - decode(center)||²`。
intrinsic center は north pole の normal coordinates で保存し、exponential map で
ambient へ復号する。同じ球面上の点なら、ambient／intrinsic の距離は一致する。

```text
||center|| <= radius * (pi - chart_margin)
```

intrinsic 表現のこの上限は、exponential map が退化する antipode 付近を除く。
既定の chart_margin は 0.05。ambient 表現では proposal を接空間へ射影して表面に戻し、
intrinsic 表現では中心座標を上の範囲へ制限する。vector state も対応する表現で輸送する。

Product／Strip の軸座標は gnomonic lift によって北半球へ写す。球面の全表面を
有限格子で覆うという意味ではない。乱数で球面点を生成する場合は backend の
`charts.construction.sphere_chart(features, intrinsic_dim=..., ...)` を使う。
compact profile の 0 点・1 点 support を防ぐには別途 covering radius を確認する。

## Torus の距離と Strip の局所性

TorusGeometrySpec は `R > r > 0` に対して `S^1 × S^(m-1)` を `R^(m+1)` に埋め込む。
普通のドーナツ表面は m=2。Line × 二次元 Grid は m=3、観測点の座標幅は4。
circle_axis は結合された幾何座標の円周成分を指定する。この成分は LinePatternSpec
から来る必要があり、Strip では分割軸と一致する。円周方向の配置は一周未満に制限する。

円周の弧長を t、残りの座標を y として、次で観測点を生成する。

```text
θ = t / R
q = (r, y) / sqrt(r² + ||y||²)
F(θ,q) = ((R+r q₀)cosθ, (R+r q₀)sinθ, r q₁, …, r qₘ₋₁)
```

q₀>0 なので Grid は断面球面の外側の一つのパッチを占める。ambient center は表面点を
m+1 個で保存する。intrinsic center は弧長 t と断面球面の normal coordinates
`v ∈ R^(m-1)` の計 m 個を保存する。t は周期 2πR で折り返し、
`||v|| <= r(pi-chart_margin)` とする。どちらも距離計算前に ambient へ復号する。

```text
a = R+r q₀, a' = R+r q'₀
D² = 4aa' sin²((θ-θ')/2) + r² ||q-q'||²
D >= 2(R-r) |sin((θ-θ')/2)|
```

この chord 距離の単位で compact profile の帯域と support 半径を指定する。
最短測地線の距離へ置き換えない。

一 tile の延長軸の幅を L、pitch を P とすると P>L で station は順序を保つ。
Euclidean では `2P-L > 2σ_max` が、離れた station へ同時に届かない条件になる。
Sphere では gnomonic lift による距離の縮みも保守的に見積もる。

Torus では station の座標範囲 `[a_i,b_i]` に対して、円の継ぎ目を含めて

```text
g_ij = min(a_j-b_i, 2πR-(b_j-a_i))
L_ij = 2(R-r) sin(g_ij/(2R))
```

を使う。非隣接 station 対で `L_ij > 2σ_max` なら同じ atom が両方へ届かない。
順序付きの disjoint interval では循環的な second-neighbor が最小候補になる。
4 station 以上では atom が届く tile は高々2。3 station では保守的な検証を行う。
2 station 以下では上限は自明である。`charts.validate_support(state, radius)` は
設定境界でこれを検証する。tile 幅だけでは曲率と円の継ぎ目を扱えない。

```python
import math
from torchcst import ChartState, geometry_presets as layout
from torchcst._backends.torch.charts import execution as charts

torus = layout.torus(3, major_radius=100 / (2 * math.pi), minor_radius=1,
                     max_arc_step=10, representation="intrinsic")
strip = ChartState(layout.strip(
    shape=(256, 784), tile_shape=(64, 784), axis=0, tile_pitch=25,
    axes=(layout.line_pattern(256, spacing=0.1),
          layout.grid_pattern((28, 28), spacing=2 / 27)), geometry=torus,
))
charts.validate_support(strip, 10)
assert strip.center_parameter_dim == 3
assert strip.embedding_dim == 4
```

max_arc_step は一更新の円周方向の弧長上限であり、support の二 tile 保証とは別。
Torch tiled、CUDA fused の routing／準備／autograd は backend に置く。

## Kernel、Optimizer、checkpoint

KernelSpec は Profile・幅・正規化・振幅・center の合成を宣言する。Geometry の
距離や制約を Kernel 側に再実装しない。CSTOptimizer は Kernel の backend に
勾配射影・proposal の適用・vector state の輸送を依頼する。atom row を直接解釈しない。

保存幅は backend の kernel.parameter_dim、自由度は kernel.parameter_dof で確認する。
CSTModule.atom_parameter_dof／cst_degrees_of_freedom も対応する値を返す。

各 State は現在の Tensor を使って実行する。declaration() は設定境界の snapshot で、
forward／backward／CUDA Graph capture 中には作らない。checkpoint は現在の buffer と
primitive metadata を保存する。異なる center 表現・shape・配置種別・学習可否を拒否し、
ロード後の固定値も検証する。旧 Geometry／Chart／Pattern クラスと旧 checkpoint の
読み替えは残していない。変更前のソースは Git revision `56e74de` にある。
