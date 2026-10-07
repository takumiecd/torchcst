# 単一ChartのPolar profile product

`CSTLinear(chart=..., kernel=presets.polar_profile_product(...))` は、一枚の
Euclidean Product/Strip Chart上で各座標軸のprofileを掛け合わせる演算を宣言する。
Chartの論理shapeは既存どおり `(out_features, in_features)`。
`composition="profile_product"`、`parameterization=PolarAmpWidthSpec(...)`、
`update.id="polar_activity_width"` は別の宣言である。

```python
import torch
from torchcst import (
    CSTLinear, CSTOptimizer, BandwidthBounds, TriweightSpec, presets,
    chart_presets as charts, pattern_presets as patterns,
)

chart = charts.product(
    shape=(16, 32),
    axes=(patterns.line(16, spacing=0.2), patterns.line(32, spacing=0.2)),
)
kernel = presets.polar_profile_product(
    profiles=(TriweightSpec(), TriweightSpec()),
    amplitude_max=1.0,
    bounds=BandwidthBounds(minimum=0.1, birth=1.0, maximum=2.0, upper_floor=0.1),
    w_c=0.1,
    radial_regularization=0.2,
)
layer = CSTLinear(chart=chart, atoms=12, kernel=kernel, backend="factored")
optimizer = CSTOptimizer(torch.optim.AdamW(layer.parameters(), lr=1e-3), model=layer)
optimizer.zero_grad()
layer(torch.randn(4, 32)).square().mean().backward()
optimizer.step()
```

## 座標とatomの所有

`profiles` はEuclidean座標順のtupleで、出力側Patternの全座標、入力側Patternの
全座標の順になる。論理軸が二つでも、入力側Patternが2D gridならprofileは三つ必要。
profile数はChartの座標次元と一致させる。形は軸ごとに異なってもよい。
各profileは幅や正規化を持たないrawなProfileBindingとして組み立てられる。

atomのParameterは `[polar_x, polar_y, center_0, ..., center_(D-1)]`。
ampと共有sigmaは先頭のPolar座標から既存のactivity-width則で得る。
一枚のChartと一つのatom中心ベクトルを保持し、計算用に入力・出力Chartを複製しない。
既存のAtomsがParameter、base optimizerがmomentsとstepを所有する。

最初の対応はEuclidean geometryとLine/Grid PatternによるProduct/Strip Chart。
Explicit/Points PatternやSphere/Torusはこの版で拒否する。これらの座標や距離を
Euclideanの独立軸と読み替えない。Stripではtile_pitchと部分端tileを実際の座標に
反映し、幅に応じた隣接stationの制限やtileごとの正規化は導入しない。

## 数学的な契約

atomを $a$、座標軸を $d$、その軸の全観測点を $i_d$ とする。
Triweightのrawな軸profileは

$$
g_{a,d}(i_d)=\left[1-
\frac{(t_d(i_d)-c_{a,d})^2}{\sigma_a^2}\right]_+^3.
$$

各軸に宣言したGaussian、Biweightなども、その既存のscalar profileを使う。
全観測点はCartesian productなので、atomのproductとノルムは

$$
P_a(i_1,\ldots,i_D)=\prod_d g_{a,d}(i_d),\qquad
n_{a,d}=\sqrt{\sum_{i_d}g_{a,d}(i_d)^2},\qquad
\|P_a\|_2=\prod_d n_{a,d}.
$$

ここでL2は、演算子の全siteを並べたベクトルのノルム（行列ならFrobeniusノルム）。
数学的な正規化は **全productに一度だけfloorを適用する**：

$$
D_a=\max\!\left(\prod_d n_{a,d},\varepsilon\right),\qquad
W_a(j,i)=\mathrm{amp}_a\frac{P_a(j,i)}{D_a},\qquad
W=\sum_a W_a.
$$

既定は `NormalizationSpec(kind="discrete_l2", domain="operator_sites", floor=1e-6)`。
`normalization=` で正のfloorを明示できる。各軸に別々のfloorは適用しない。
たとえば各軸ノルム0.6、floor0.5なら、積0.36にfloor0.5が発動する。
空支持はゼロの寄与を返す。正の単一siteでもfloorが発動する場合はampそのものに
置き換えない。floor非活性なら各軸のL2正規化を掛け合わせる計算と一致する。

## Hによる参照計算と微分

backendは出力側・入力側のrawな軸productを $U_a(j),V_a(i)$ にまとめ、

$$
H_{b,a}=\sum_i X_{b,i}V_a(i),\qquad
Y_{b,j}=\sum_a H_{b,a}\frac{\mathrm{amp}_a U_a(j)}{D_a}
$$

を実行できる。ノルム計算のために全演算子のsite表や全raw productを展開しない。
Torch `factored` は入力・出力factorとHを保持する参照方式、`materialized` はWを
生成する参照方式。全係数は現在のatomとChart bufferから毎回評価する。
`operator.factors()` も単一Chartに対して利用できる。

既存Polarと同様、task微分では現在のsigmaを定数として扱う。
amp・全中心・正規化の分母に関する微分は保持する。sigmaは固定設定ではなく、
各stepのPolar更新後に変化する。既存のfinite_chord/time_energy、radial regularizer、
activity_gain、dormant_expansion_rateを同じ更新則で使う。

旧separableは入力・出力ごとにfloorを適用し、中心は入力・出力の順に保存する。
新方式は全productにfloorを適用し、中心は単一Chartの出力・入力座標順に保存する。
checkpointや宣言を旧方式から暗黙に読み替えない。新しい正規化の契約もcheckpointに記録する。

## 検証と実装範囲

`tests/test_profile_product.py` は全siteを列挙する独立FP64 oracleでY、dX、全source勾配、
異なる軸profile、3D座標、両側のStrip pitch、部分端tile、空/単一支持、global floorを検証する。
既存separable Polarと対応する中心順を明示してAdamWのParameter、moments、step、
変化する幅を比較し、weights-only checkpointも確認する。

この追加は数学的な宣言とTorch参照実装。CUDA local_productへの接続、Strip用の
外側スケジュール、大型GPUの完全step時間・capture/replayピークは別の検証対象。
既存radial、separable、strip_torusの意味と既定recipeは維持する。
