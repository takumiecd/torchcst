# Chart geometry contract

`Chart`は単なるfeatureラベルではなく、観測siteとatom centerが存在する
幾何を所有する。Kernelはその幾何の上でbasisを構成するが、距離や領域の
制約を再実装しない。

## 責務

### Geometry

- intrinsic dimension、site embedding dimension、center parameter dimension
- site/center間の距離
- centerの初期化
- tangent projection
- retraction
- vector-like optimizer stateのtransport

`EuclideanGeometry(d)`では両dimensionが`d`であり、更新は加算である。
`SphereGeometry(d)`は`S^d`を`R^(d+1)`へ埋め込み、chord距離を使う。
ambient表現では更新proposalは接空間へ射影され、球面へnormalizeされる。
intrinsic表現では中心だけをnorth poleまわりのnormal coordinates `R^d`で
保存し、距離計算時にexponential mapで`R^(d+1)`へ復号する。

### Chart

- fixed-cardinalityな観測site
- 論理的なテンソル形状`shape`と、各軸に対応する`SitePattern`
- Geometryの所有
- spacingなどsite配置由来のmetadata

Chartはprofile形状、bandwidth、amplitude、atom数を知らない。
`ProductChart`は`shape`の各軸に対応する`axes`を直積的に解釈する。
テンソルの階数は`len(shape)`、幾何座標の次元は各patternの`dim`の和であり、
両者は同じでなくてよい。例えば`shape=(64, 784, 4)`に対し、
`axes=(LinePattern(64), GridPattern((28, 28)), LinePattern(4))`なら
三階テンソルを四次元の幾何に配置する。`CSTLinear`だけが重みを
`[out, in]`に制限する。ChartのAPIには入出力を表す特別な軸名はない。

`StripChart`も同じ`shape`と`axes`を受け取る。`axis`で指定する延長軸は
`LinePattern`でなければならず、それ以外の軸はタイル分割しない。
`tile_shape`は各軸のタイル幅を表し、`tile_pitch`は延長軸上の隣接タイルの
座標間隔となる。タイル列のために新しい幾何座標軸は追加しない。
両Chartとも必要なsite番号の座標だけを生成し、テンソル全体の座標表を
保持しない。

```python
ProductChart(shape=(64, 784), axes=(axis0_line, axis1_grid))
StripChart(
    shape=(64, 784), axes=(axis0_line, axis1_grid),
    tile_shape=(8, 784), axis=0, tile_pitch=4.1,
)
```

`LinePattern`、`GridPattern`、`PointsPattern`はsiteの局所的な並べ方を
表す。`spacing`とGeometryが距離の意味を決める。`low/high`はgridの
両端を直接指定したい場合の代替である。

新しいStrip配置で一タイル内の延長軸の幅を`L`、compact kernelの最大半径を
`R`、`tile_pitch`を`P`とすると、`P > L`でタイルの順序が保たれる。
さらに`2P - L > 2R`なら、離れた二タイルへ同時に届かないため、
一つのatomが届くタイルは最大2個になる。`tile_indices(station)`は
小さなタイル内の対応だけを計算し、全siteのID表を保持しない。
`packed_weight()`はstation順の物理配列を返すが、現行のPyTorch forwardは
密な行列を使う。SphereGeometryを指定した場合は、各軸の直積座標を
北半球へ単射で写し、centerは球面上で更新する。Sphereの支持範囲検証では
この写像による距離の縮みを保守的に見積もる。

## TorusGeometryとStripの局所性

`TorusGeometry(m, major_radius=R, minor_radius=r)`は、`R>r>0`のとき
自由度`m`の輪状超曲面 `S^1 × S^(m-1)` を `R^(m+1)` に埋め込む。
普通のドーナツ表面は`m=2`であり、`LinePattern × GridPattern((28,28))`
なら`m=3`、siteとambient centerの保存幅は4になる。
`circle_axis`は結合された幾何座標のどの成分を円周方向に使うかを指定する。
その成分は`LinePattern`から来なければならず、Stripでは分割軸と一致する。

円周軸の線座標を`t`、残りの`m-1`個の座標を`y`として、
`θ=t/R`、`q=(r,y)/sqrt(r²+||y||²) ∈ S^(m-1)`とする。
Chartは要求されたsiteだけについて次の位置を作る。

```text
F(θ,q) = ((R+r q₀)cosθ, (R+r q₀)sinθ, r q₁, …, r qₘ₋₁)
```

`q₀>0`なので、`GridPattern`のsiteは断面球面の外側の一つのパッチを
占める。幾何のcenterは輪状超曲面全体を動ける。compact profileでは
観測siteのない領域へcenterを初期化するとsupportが消えるため、
通常は`atom_init="balanced"`を使う。単一Gridで断面球面の全面を
覆ったとは解釈しない。

距離はSphereと同じambient chord距離で、断面を`q,q'`、円周角を
`θ,θ'`、`a=R+r q₀`、`a'=R+r q'₀`と置くと

```text
D² = 4aa' sin²((θ-θ')/2) + r² ||q-q'||²
   = ||F(θ,q)||² + ||F(θ',q')||² - 2 F(θ,q)·F(θ',q')
D  ≥ 2(R-r) |sin((θ-θ')/2)|
```

である。最短測地線を数値的に解く必要はない。最後の等式は将来の
GEMM評価にも使える。`Triweight`などのcompact Profileのsupport半径
`σ_max`もこのchord距離の単位で指定する。

局所性の検証では、各タイルの円周軸上の座標範囲`[a_i,b_i]`を
保持せずに計算する。`i<j`の二タイル間の最小円周方向gapを

```text
g_ij = min(a_j-b_i, 2πR-(b_j-a_i))
L_ij = 2(R-r) sin(g_ij/(2R))
```

とすると、任意の両タイルのsite間距離は`L_ij`以上になる。
タイル列は一周未満に制限する。隣接タイルは円の継ぎ目を含めて
循環的に定義する。**隣接しないすべてのタイル対で
`L_ij > 2σ_max`**なら、三角不等式から同じatomがその対の両方に
届くことはない。4タイル以上では、任意の三タイルに非隣接対が
含まれるため、atomが届くタイル数は高々2となる。
2タイル以下ならこの上限は自明である。3タイルでは全対を検証する
保守的な条件を用いる。`StripChart.validate_support`はこれを
事前に判定し、満たさない設定を拒否する。
**タイル幅が`2σ_max`を超えるだけでは、曲率と円の継ぎ目を
考慮できないため十分ではない。**

`max_arc_step`を指定すると、retraction時の円周方向の移動を
一更新あたりその長さ以下に制限できる。`tile_pitch`より小さくすれば
タイルstationを何個も飛び越す更新を避けられる。これは
supportの二タイル保証とは別の制約である。現在の`packed_weight()`は
タイル順の密な配列を作るが、atomの局所的な物理再配置やnativeな
タイル演算はまだ実装していない。

```python
import math
from torchcst import GridPattern, LinePattern, StripChart, TorusGeometry

torus = TorusGeometry(
    3, major_radius=100 / (2 * math.pi), minor_radius=1,
    max_arc_step=10,
)
strip = StripChart(
    shape=(256, 784), tile_shape=(64, 784),
    axes=(LinePattern(256, spacing=0.1),
          GridPattern((28, 28), spacing=2 / 27)),
    axis=0, tile_pitch=25, geometry=torus,
)
strip.validate_support(10)
```

### Profile

- Chartが返す距離へ適用するscalar shape
- Gaussian/Triweight/Wendlandなどの評価
- 離散L2正規化
- center sliceの初期化、gradient projection、retraction、transportを
  Chart Geometryへ委譲

### Kernel

- 完全なatom row `p`のレイアウト
- input/output Profileの結合
- amplitude/bandwidthなどKernel固有座標
- stored parameter widthとintrinsic degrees of freedomの集計

単一チャートの`DirectAmpWidth`は`[w, q, center...]`という一行の
パラメータから直接重みへの寄与を計算する。入力・出力のfactorは使わない。
`w`は符号付き振幅、`q`は振幅更新から進む帯域activityで、Profileは
Triweightなどから選ぶ。単一チャートでは正規化しないProfileの
site slice評価を使い、全site×全atomの距離テンソルを保持しない。
StripChartの二タイル制約には設定された最大半径`sigma_max`を使う。

例えばPolar Kernelのparameter spaceは

```text
Polar coordinates × input Chart geometry × output Chart geometry
```

という積空間になる。

### Optimizer

- Kernelへparameter gradientの接空間射影を依頼
- optimizer proposalを作成
- Kernelへretractionを依頼
- vector first momentを新しい接空間へtransport

optimizerはatom rowの列を直接解釈しない。

## Storageと自由度

`S^3`上のcenterには二つの保存方式がある。

- `representation="ambient"`（既定）: `R^4`の単位vectorとして4個保存。
  冗長だがglobalで安定する。
- `representation="intrinsic"`: normal coordinatesとして3個保存。
  north poleの反対側にあるantipode近傍だけをchartから除外する。

どちらもintrinsic degrees of freedomは3である。モデル予算の比較には
`kernel.parameter_dof(...)`、実際の保存量には`kernel.parameter_dim(...)`を使う。

`CSTModule.atom_parameter_dof`はatom一個の自由度、
`CSTModule.cst_degrees_of_freedom`はsite全体の自由度を返す。

## 現在のSphere contract

```python
from torchcst import Chart

chart = Chart.sphere(features=64, intrinsic_dim=3)
assert chart.coordinates.shape == (64, 4)
assert chart.intrinsic_dim == 3
assert chart.embedding_dim == 4
assert chart.center_parameter_dim == 4

compact = Chart.sphere(
    features=64,
    intrinsic_dim=3,
    representation="intrinsic",
)
assert compact.coordinates.shape == (64, 4)
assert compact.center_parameter_dim == 3
```

siteは両方式とも球面上の`d+1`次元座標である。intrinsic版で`p`に保存する
centerだけが`d`次元になり、`Geometry.decode_centers`を通して球面上へ戻される。
したがって同じ球面上の点なら、ambient版とintrinsic版のKernel距離は一致する。

intrinsic版の有効範囲は

```text
||p_center|| <= radius * (pi - chart_margin)
```

であり、更新時にこの範囲へretractされる。既定の`chart_margin=0.05`は
exponential mapが退化するantipodeを避けるための角度marginである。この方式は
1 scalar/centerを削減する代わりに、小さなantipodal capを中心の到達領域から除く。

`Chart.sphere`は球面上のsiteを生成する。これは領域外へのcenter escapeを
防ぐが、有限site集合のcoveringを自動保証するものではない。compact
profileで0点・1点supportを防ぐには、別途第2近傍covering radiusを測り、
`sigma_min`をそれより大きく選ぶ必要がある。

## 後方互換性

`Chart.points`、`Chart.linspace`、`Chart.grid`はgeometry未指定時に
`EuclideanGeometry`を生成する。従来の距離、初期化、parameter storage、
forward、微分、更新結果はそのまま維持される。
