# 等間隔の平坦なTorus chart

`chart_presets.periodic_grid` は、独立した周期軸を持つ単一chartを宣言する。
既存の3D埋め込みTorusとは距離が異なるため、Geometry IDは`flat_torus`、
chart kindは`periodic_grid`、対応するprofile productはkernel revision 3とする。

```python
import torch
from torchcst import (
    BandwidthBounds, CSTLinear, TriweightSpec,
    chart_presets as charts, presets,
)

# 出力16*32=512、入力64。座標は3次元、行列は[512, 64]。
chart = charts.periodic_grid(
    (16, 32, 64), periods=(1.0, 2.0, 4.0),
    origin=(0.0, -0.2, 0.1), output_dims=2,
)
kernel = presets.polar_periodic_profile_product(
    profiles=(TriweightSpec(),) * 3,
    amplitude_max=1.0, w_c=0.1,
    bounds=BandwidthBounds(minimum=0.02, birth=0.1,
                           maximum=0.4, upper_floor=0.05),
)
model = CSTLinear(chart=chart, atoms=32, kernel=kernel,
                  dtype=torch.float64, backend="factored")
y = model(torch.randn(8, 64, dtype=torch.float64))
```

## 配置と状態

D軸に正の整数点数`grid_shape`、正の有限な`periods`、有限な`origin`を指定する。
`origin`の省略値はゼロ。D≥2、`1 <= output_dims < D`であり、先頭の
`output_dims`軸を出力、残りを入力にする。各側は最後の軸が最速のrow-major。
3次元以上も同じ宣言/実装を使い、Dの上限を特別に固定しない。

\[
h_d=L_d/n_d,\quad t_d(k)=o_d+kh_d\;(0\le k<n_d),\qquad
N_o=\prod_{d<p}n_d,\quad N_i=\prod_{d\ge p}n_d.
\]

終端点を重複させず、各軸は円を一周する。軸ごとにperiod/spacingは異なってよい。
`PeriodicGridChartSpec.shape`は行列の`(N_o,N_i)`で、`grid_shape`は座標のD軸。
`PeriodicGridChartState.spacing`はlive periods/countsから導出する。
固定bufferは原点と周期の計2D scalarのみで、site座標表・PatternState・index表を
保持しない。点数と分割は静的な整数metadata。必要なsiteだけ番号から生成できる。
FP16等では格子座標が丸められるため、隣接点の数値的な識別まで保証するものではない。

原点/周期は`.to()`に追従し、checkpointに保存される。点数/分割が異なる
checkpointは、行列shapeが同じでも拒否する。load時に有限性と正の周期を検証する。
原点/周期を変える場合はeagerの設定境界で行う。siteは学習Parameterではない。

## 数学と微分

\[
\mathbb T^D=\prod_d\mathbb R/L_d\mathbb Z,\qquad
\delta_L(z)=z-L\lfloor z/L+1/2\rfloor\in[-L/2,L/2).
\]

各軸のprofileに最短周期差の二乗を渡す。Triweightなら

\[
g_{ad}(k)=\big[1-(\delta_{L_d}(t_d(k)-c_{ad})/\sigma_a)^2\big]_+^3,
\quad P_a(\mathbf k)=\prod_d g_{ad}(k_d).
\]

Cartesian配置によりfull-domain離散L2 normが厳密に分解される。

\[
\nu_a=\prod_d\sqrt{\sum_k g_{ad}(k)^2},\qquad
W_{ji}=\sum_a A_aP_a(j,i)/\max(\nu_a,\epsilon).
\]

floorは全積の後に一度だけ適用する。軸ごとのfloorではない。
Gaussianなど既存profileも指定可能だが、無限支持profileは有限区間探索の対象外。
半周期以上の幅も許し、端点や支持を切り捨てない。

atomのParameterは`[K,D+2]`で、最初の2値が既存Polar座標、残りが中心。
振幅・共有幅・activity updateの既存則を使い、task VJPで幅はstop-gradient。
幅はoptimizerによるactivity更新で変わる。中心と振幅にはnorm微分も含める。
中心更新は`remainder(center + displacement, periods)`で`[0,L)`へwrapする。
chart原点と異なる代表座標を使っても距離は不変。接空間のmomentはwrapで回転しない。

反対点では最短距離が非滑らかになる。ちょうど半周期の差は負の半周期へ写し、
Torch VJPはこの枝の片側微分を使う。反対点で数学的に滑らかだとは扱わない。
通常点の有限差分検証と、この枝の規約検証は別に行う。

## 今回使える計算方式と次の比較

現時点で動くのはTorchの`factored`、`materialized`と既存`auto`の参照経路。
chartのO(D)は固定配置の保存量であり、学習全体のメモリではない。
以下は同じWを計算する方式の比較で、速度順ではない。
Kはatom数、Bはbatch数、Cは一度に処理するatom数、R/Tは行列窓の行/列数。

| 方式 | 主な利点 | 費用・弱点 | 今回の役割 |
| --- | --- | --- | --- |
| 全atomを行列にして合計 | 演算が直接的で独立oracleと照合しやすい | 現行materializedは一時的にK×No×Niを作り、大型用には重い | 小格子の正しさ参照 |
| 軸productをinput/output factorへ展開、2回GEMM | W/dWを不要にできる。Y=(X Fi) Foᵀ | factorはK(No+Ni)、中間H/GはBK。支持ゼロも密に計算 | 実装済みfactored基準。Kが小さい場合の有力候補 |
| 支持範囲からgrouped Wを組み立てGEMM | 過去のgrouped patch/GEMMを再利用し、X/dYとの不規則アクセスを減らせる | W/dWはNoNi。狭い支持でも大きなWの読み書きが残る | 最初のCUDA比較基準候補 |
| bounded W窓＋GEMM | Wの一時領域をRTに抑えつつGEMMを維持 | 窓ごとの準備・再読込・backward累積。候補index/routingにも費用 | Wの容量が支配した場合の優先候補 |
| bounded atom/H-G contraction | factorはC(No+Ni)、H/GはBCへ制限可能 | chunk分割でGEMMが小さくなり、X/dYを繰り返し読む場合がある | 広い支持・W不要のメモリ候補 |
| 支持内のsiteだけ直接contract | 狭い支持でゼロ演算とW保存を減らせる | gather/scatter、atomic、X/dY再読込、batch利用効率。保存量減が速度改善とは限らない | 前方式の測定後に比較。CSRを前提にしない |
| D軸のtensor contraction | 軸factorだけならK∑ndで保存でき、flatten factorも不要にできる | input全体の読み出しは残る。中間Tensor、軸順、strideで費用が変わる | 3次元以上でfactor容量が支配した場合の候補 |

支持が小さいTriweightでは、各軸のstart/countをsigma・spacing・中心から求め、
周期番号を生成できる。範囲metadataはO(KD)で、全site IDのCSRは不要。
ただし、軸準備は∑mdでも、Wへの支持寄与は∏md。Dが増えるとこの差が大きくなる。
この算術支持計算とCUDA実装は次の段階であり、今回のTorch参照は各軸全点を評価する。

**次の方針は、全量Torch参照とgrouped W＋GEMMから同条件の比較を始めること。**
prep/norm・W組立・forward/dX/dW・atom VJP・optimizerの時間を診断し、
未計装の完全step時間とallocated/reservedピークを別途測る。
W/dWが支配すればbounded Wまたはbounded H/G、支持処理が支配すれば
算術範囲と軸factor再利用、転送が支配すれば連続アクセス・共有・配置を比較する。

最内側の格子軸を連続アクセスに利用する。atom順序の変更は候補であり、
順序変更だけでcache効率が上がるとは仮定しない。warp内の隣接アクセスと
読込み再利用をprofilerで確認する ([NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory))。
任意の独立atom中心と幅を持つこの演算は一般には平行移動不変な畳み込みではなく、
FFTをそのまま適用する前提はない。

過去の[grouped matrix実測](research-history/local-product/20261008-profile-product-grouped-matrix.md)、
[matrix-free比較](research-history/local-product/20261010-product-matrix-free-results.md)は
方式選定の参考である。旧Euclidean/Sphere/embedded Torusの時間を新chartへ転用しない。

## 検証範囲

`tests/test_periodic_grid.py`はD2/3/4、非対称period/origin、入力/出力分割、
singleton軸、周期外中心、seam、空/単一/広い支持、全積floorを検証する。
独立oracleは全格子を列挙し、`min(remainder, period-remainder)`で距離を計算し、
全積を一度に正規化する。Y、dX、全atom勾配をFP64で照合する。
通常点の中心有限差分、反対点のVJP枝、周期移動不変性、indexed positions、
O(D)固定buffer、live設定/checkpoint、dtype、Adam seam更新・moment・clock・live幅も確認する。

参照テスト60件、canonical CPU suiteは1466 passed / 2644 skipped。
wheel/sdist buildと上記の利用例も確認した。GPU・実DBを必要とするskipは未検証。

CUDAの正しさ、Graph capture/replay、GPUの速度・ピークメモリは未測定。
新しいCUDA Algorithm/recipe、benchmark fixture、dispatcherへの性能採用は別段階。
