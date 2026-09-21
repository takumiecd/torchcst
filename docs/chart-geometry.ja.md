# Chart geometry contract

`Chart`は単なるfeatureラベルではなく、観測siteとatom centerが存在する
幾何を所有する。Kernelはその幾何の上でbasisを構成するが、距離や領域の
制約を再実装しない。

## 責務

### Geometry

- intrinsic dimensionとembedding dimension
- site/center間の距離
- centerの初期化
- tangent projection
- retraction
- vector-like optimizer stateのtransport

`EuclideanGeometry(d)`では両dimensionが`d`であり、更新は加算である。
`SphereGeometry(d)`は`S^d`を`R^(d+1)`へ埋め込み、chord距離を使う。
更新proposalは接空間へ射影され、球面へnormalizeされる。

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

`S^3`上のcenterは`R^4`の単位vectorとして保存する。したがってstorage
widthは4だがintrinsic degrees of freedomは3である。モデル予算の比較には
storage tensorの要素数ではなく`kernel.parameter_dof(...)`を使う。

`CSTModule.atom_parameter_dof`はatom一個の自由度、
`CSTModule.cst_degrees_of_freedom`はsite全体の自由度を返す。

## 現在のSphere contract

```python
from torchcst import Chart

chart = Chart.sphere(features=64, intrinsic_dim=3)
assert chart.coordinates.shape == (64, 4)
assert chart.intrinsic_dim == 3
assert chart.embedding_dim == 4
```

`Chart.sphere`は球面上のsiteを生成する。これは領域外へのcenter escapeを
防ぐが、有限site集合のcoveringを自動保証するものではない。compact
profileで0点・1点supportを防ぐには、別途第2近傍covering radiusを測り、
`sigma_min`をそれより大きく選ぶ必要がある。

## 後方互換性

`Chart.points`、`Chart.linspace`、`Chart.grid`はgeometry未指定時に
`EuclideanGeometry`を生成する。従来の距離、初期化、parameter storage、
forward、微分、更新結果はそのまま維持される。
