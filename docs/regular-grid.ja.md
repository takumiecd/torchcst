# weight軸ごとに宣言する等間隔格子

`chart_presets.regular_grid`は、格子の点配置とgeometryを分けて宣言する。
点数・間隔・原点から必要な座標だけを生成し、全siteの座標表やindex表を保存しない。

```python
import torch
from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, presets
from torchcst import chart_presets as charts

chart = charts.regular_grid(
    grid_shape=((16, 16), (128,)),
    spacing=0.1,
    geometry="flat_torus",
)
assert chart.shape == (256, 128)
assert chart.geometry.periods == (1.6, 1.6, 12.8)

kernel = presets.polar_periodic_profile_product(
    profiles=(TriweightSpec(),) * 3,
    amplitude_max=1.0,
    w_c=0.1,
    bounds=BandwidthBounds(minimum=0.02, birth=0.1, maximum=0.4, upper_floor=0.05),
)
model = CSTLinear(chart=chart, atoms=8, kernel=kernel, backend="factored")
y = model(torch.randn(4, 128))
assert y.shape == (4, 256)
```

## 格子とweightの対応

`grid_shape`の各tupleがweightの一軸に対応する。
`((16,16),(128,))`なら、weight第0軸は2次元格子の256点、第1軸は1次元の128点。
LinearではWは`[out,in]`なので、この例の出力側は2次元、入力側は1次元である。
入力側を2次元にした同じweight shapeには`((256,),(16,8))`を使う。

\[
N_r=\prod_{d\in G_r} n_d,\qquad
\mathrm{shape}=(N_0,\ldots,N_{R-1}),\qquad D=\sum_r |G_r|.
\]

weight shapeと次元数を重複して指定しない。`output_dims`も不要。
座標の順序は各tupleを先頭から連結した順序で、最後の座標軸が最速。
各weight軸内の格子も最後の座標軸が最速のrow-majorでflattenする。
Chart単体は任意の正の次元数・weight軸数を宣言できる。
`CSTLinear`とprofile-product実行にはweight軸が二つ必要。

## spacingとgeometry

chartの間隔の名前は`spacing`に統一する。
scalarはD座標軸すべてへbroadcastし、tuple/listはD個の値を指定する。
`spacing`は有限の正値、既定値は1.0。
`origin`もscalarかD個の値で、既定値は0.0。originは格子の最初の点でありatomの中心ではない。

\[
t_d(k)=o_d+k h_d,\qquad 0\le k<n_d.
\]

`geometry`を省略するか`"euclidean"`にするとD次元Euclidean。
`"flat_torus"`にすると各周期を`L_d=n_d h_d`から導出し、終端点を重複せず一周する。
同じscalar spacingでも点数が異なる軸ではperiodが異なる。
spacingは隣接点の間隔、geometryのperiodsは一周の長さであり、意味を混同しない。

具体的な`EuclideanGeometrySpec`または`FlatTorusGeometrySpec`も指定でき、
次元数がDと一致することを検証する。具体geometryのperiodsは上書きしない。
その場合、格子spacingとgeometryのperiodsは独立した指定で、一周の全格子とは限らない。
このCartesian配置に対するTorchのprofile-productはそのまま正規化・距離を計算する。
Sphereや埋め込みTorusには異なる座標写像が必要なので、このchartでは受け付けない。

geometryの選択はchartの構造から分離する。Euclideanには
`presets.polar_profile_product`〈revision 1〉、FlatTorusには
`presets.polar_periodic_profile_product`〈revision 3〉を使う。
profileはD個必要。距離の異なるrevisionを取り違えた組み合わせは拒否する。
FlatTorusの中心更新は周期でwrapし、Euclideanの中心更新は既存のEuclidean則を使う。

## 正規化と状態

座標ごとのprofile積を使い、full-domain離散L2 normの全積の後にfloorを一度だけ適用する。
Cartesian格子なので因数分解しても同じWを表す。軸ごとのfloorへ置き換えない。
既存Polarの共有幅、task-width stop-gradient、振幅・中心の微分とactivity更新を維持する。

Stateの点配置bufferはorigin/spacingの計2D個。FlatTorusではgeometryがperiodsのD個を持つ。
座標軸vectorは必要時に生成する。O(D)は固定配置の保存量であり、学習全体のメモリではない。
低精度では隣接点が丸められるため、すべての点の数値的な識別は保証しない。

origin/spacing/periodsはdevice・dtype・checkpointに追従する。
固定配置の値を変更する場合はeagerの設定境界で行う。checkpointは具体型と格子の対応を検査し、
weight shapeが同じでも異なるgrid_shapeは拒否する。非有限origin、非正spacing/periodsを拒否する。

## 検証と実行範囲

`tests/test_regular_grid.py`でscalar/軸別指定、1次元から4次元の座標、
非対称のweight軸対応、独立全格子oracleによるY/dX/全atom勾配、空支持・singleton・
広い支持・全積floor、checkpoint、dtype、geometryに応じた公開optimizer更新を検証する。
Torchのfactored/materialized/autoを接続する。
既存`PeriodicGridChartSpec`とそのcheckpoint・研究CUDA経路は維持する。
新RegularGridChart向けのCUDA Algorithm拡張と速度・peak memoryの計測は後続作業である。
