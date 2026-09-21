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
- Geometryの所有
- spacingなどsite配置由来のmetadata

Chartはprofile形状、bandwidth、amplitude、atom数を知らない。

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
