# 2次B-spline profileの定義とTorch参照

`QuadraticBSplineSpec()`は、ピーク1の対称な一つの山を宣言する。
中心と幅で移動・伸縮する3区間の2次多項式で、値と1階微分が連続なC1関数。
2階微分は区間のつなぎ目で変わる。
2026-10-17にTorch参照を追加し、累積集計の最適化に使う形を固定した。
既存profileの定義、checkpointを置き換えない。

## 距離と幅

1次元の距離を `d=x-mu`、支持半幅を `R=sqrt(6)*sigma`、`u=d/R` とすると、

\[
F_\sigma(d)=\begin{cases}
\frac32(1+u)^2,&-1<u<-\frac13,\\
1-3u^2,&-\frac13\le u\le\frac13,\\
\frac32(1-u)^2,&\frac13<u<1,\\
0,&\text{otherwise}.
\end{cases}
\]

左・中央・右は別の多項式で、境界は `mu-R, mu-R/3, mu+R/3, mu+R`。
中心は1、内部のつなぎ目は2/3、支持端は0。支持端の傾きも0になる。

\[
-F_\sigma''(0)/F_\sigma(0)=1/\sigma^2.
\]

sigmaはこの中心曲率幅で、Gaussianの標準偏差や分布の実際の標準偏差とは区別する。
既存`TriweightSpec()`は支持半幅sigmaなので、その幅を暗黙に読み替えない。

宣言IDは`quadratic_bspline`、profile revisionは1。通常のscalar profileと同様、
入力は `q=squared_distance*precision`、`precision=1/sigma²`：

\[
f(q)=\begin{cases}
1-q/2,&0\le q\le2/3,\\
\frac32(1-\sqrt{q/6})^2,&2/3<q<6,\\
0,&q\ge6.
\end{cases}
\]

Profileは形だけを持つ。幅はparameterization、距離はChart/Geometry、正規化は
NormalizationSpecが定める。多次元radialとして使う場合は距離にこの関数を適用し、
座標別のB-spline積とは別の合成になる。

## 微分と正規化

外側で `r=sqrt(q/6)` と書くと `df/dq=-(1-r)/(4r)`、中央では-1/2、支持外では0。
Euclidean中心の微分は `-2*precision*(df/dq)*(x-mu)`、
precision微分は `squared_distance*(df/dq)`。
Torch実装は使わない中央側の平方根入力を下限1/9で保護し、
q=0で`0*inf`を作らない。中央の式そのものにepsilonは加えず、peakと曲率を保つ。
境界では選択した枝の微分を使い、値と1階微分は左右で一致する。

浮動小数点のsite座標で`(sqrt(6)*sigma)²*precision`が6より僅かに小さくなる場合は、
支持内の微小な値を保つ。数値誤差を隠すための支持の丸めや非零値の切り捨てを行わない。
引数qが6以上なら、値と1階微分は0。

単独の`fixed_profile(..., normalize=True)`は既存compact profileと同じchart-site L2と
floor1e-6を使う。profile-productでは各軸をrawとして、

\[
Q_{a,d}=\sum_{i_d} F_{a,d}(i_d)^2,\quad
D_a=\max\left(\sqrt{\prod_d Q_{a,d}},\varepsilon\right),\quad
W_a=\frac{\mathrm{amp}_a}{D_a}\prod_d F_{a,d}
\]

を評価する。floorは全積へ一度だけ適用し、離散normの中心微分も保持する。
Polarのtask VJPでは現在のsigmaを固定し、幅は既存のPolar更新則で変化する。
単独profileのprecision微分は利用できる。幅更新則自体は変更していない。

## 使用例

```python
import torch
from torchcst import (
    BandwidthBounds, CSTLinear, QuadraticBSplineSpec,
    chart_presets, presets,
)

kernel = presets.polar_profile_product(
    profiles=(QuadraticBSplineSpec(),) * 2,
    amplitude_max=1.0,
    bounds=BandwidthBounds(
        minimum=0.1, birth=0.25, maximum=12.8, upper_floor=0.1,
    ),
    w_c=0.1,
)
layer = CSTLinear(
    chart=chart_presets.regular_grid(((32,), (64,)), spacing=0.1),
    atoms=8, kernel=kernel, backend="factored", dtype=torch.float64,
)
x = torch.randn(4, 64, dtype=torch.float64, requires_grad=True)
layer(x).square().mean().backward()
```

最短周期座標では、同じprofilesを`polar_periodic_profile_product`へ渡す。
Torch `materialized`／`factored`／`auto`の既存参照経路で実行できる。
一般的なGeometryでの距離評価と、1次元座標での区分多項式集計は区別する。
累積集計の性質を使う対象はまずEuclidean／最短周期変位のprofile-product。

## 実装と検証

- 宣言：`src/torchcst/kernels/profiles/bspline.py`。
- Torch値／解析微分：`src/torchcst/_backends/torch/profiles/compact.py`。
- 検証：`tests/test_quadratic_bspline_profile.py`。

独立した符号付き3区間式、全siteのprofile-productと全演算子L2をoracleにする。
中心／内部knots／支持端、空支持、norm floor、解析中心／precision微分、
RegularGridの複数座標、周期seam、可変sigma、Y/dX/全Polar source勾配、
AdamW＋Polarの5更新とmoments、checkpoint識別を照合した。
FP64参照とFP32出力・勾配を比較し、FP16/BF16の有限性も検証している。

Torch参照はsiteごとに正しい値を作る経路。共有prefix／局所離散L2の最適化は
この参照と照合する別Algorithmとして扱い、runtimeへはまだ接続していない。
CUDA最適化、Graph、完全step速度とGPUピーク容量は次の検証対象。
CPU suiteのGPU／DB skipを実機検証には数えない。
