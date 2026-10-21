# σでLを選ぶ有限Fourier profileとTorch参照

`DyadicFourierSpec(modes=4 または 8)`は、以前の研究用σ依存Lの選択を
小さい尺度まで使うための宣言である。新しい山を別途作るのではなく、
[bottom-up CPU研究](research-history/cuda-linear/bottom-up-fourier-cpu-20261021.ja.md)
で照合したFourier-only関数をruntimeから評価できるように整理した。
中心・σは連続値のまま、Lだけを2倍刻みで選び、移行中は隣接尺度を混ぜる。
多項式の直接評価、追加の空間窓、部分タイルの切り捨ては含めない。

公開済み`ChebyshevSquareSpec`等の「固定周期・3タイル支持」とは定義が異なる。
それらの設定やcheckpointを読み替えず、profile ID `dyadic_fourier` revision 1で識別する。

## 尺度と関数

物理変位をr=x−μとし、基本の半幅をL_minとする。

\[
L_j=L_{\min}2^j,\quad j\ge0,\qquad R=\sqrt{11}\sigma.
\]

各成分は支持範囲[-L_j,L_j]、基本周期2L_jで、支持外は厳密に0。
L_minは関数の定義の一部であり、実行時タイルの大きさを変更しても変えない。
格子と整合する場合は例えば`L_min=4*spacing`と宣言する。
spacing=.1ならL候補は.4,.8,1.6,3.2…となる。
実行タイルの細分化・親子関係・使用周波数集合は別のAlgorithmで決める。

\[
F_{L,\sigma}(r)=\mathbf1_{|r|<L}\sum_{k=0}^{M}b_k(\sigma,L)
 \cos(\pi kr/L).
\]

係数の目標関数はP(r)=(1−r²/R²)^4_+。sigma²はPの連続L1分散だが、
制約付き有限Fourier関数の分散や中心曲率が厳密にsigma²になる契約ではない。
Pは係数を決めるためだけに用い、小さいsigmaでも実際の評価はFourierで行う。
M=4/8では形の近似精度は異なり、小さい負の裾を許す。非負性は保証しない。

未制約cosine係数を

\[
\tilde b_0=\frac{R}{2L}\int_{-1}^{1}(1-u^2)^4du,\quad
\tilde b_k=\frac{R}{L}\int_{-1}^{1}(1-u^2)^4
 \cos(\pi kRu/L)du\ (k>0)
\]

とする。実装はこの積分の解析式を小引数の安定な級数と併用する。
W=diag(2,1,…,1)、制約Aの行を
((-1)^k(k/M)^(2i))_kと全要素1の行とし、目標t=(0,…,0,1)とすると、

\[
b=\tilde b+W^{-1}A^T(AW^{-1}A^T)^{-1}(t-A\tilde b).
\]

M=4ではi=0,1、M=8ではi=0,1,2を使う。
中心値1を保ち、支持端でそれぞれF,F''またはF,F'',F''''を0にする。
偶対称cosine級数の奇数階微分も支持端で0なので、ゼロ拡張はそれぞれC3/C5。
これは連続L2誤差の制約付き射影であり、離散site L2正規化とは別処理。

## σによる移行

既定値ρ=`level_ratio`=.85、h=`overlap`=.12とする。
RがB_j=ρL_jの[(1−h)B_j,(1+h)B_j]に入ると、

\[
z=\frac{R-(1-h)B_j}{2hB_j},\quad
w(z)=6z^5-15z^4+10z^3,\quad
F=(1-w)F_{L_j,\sigma}+wF_{2L_j,\sigma}.
\]

それ以外はj=max(0,ceil(log2(R/(ρL_min))))の単一成分。
移行区間の両端でwの1階・2階微分が0となり、sigma方向にC2で接続する。
整数routingそのものは微分せず、係数とwのsigma微分を両方保持する。
h<1/3で移行区間の重複を避け、ρ(1+h)<1で目標Pの支持を各成分内に保つ。
最小Lは下限なので、sigmaを際限なく小さくしても形を保てる保証はない。

## Torch参照と正規化

宣言は`kernels/profiles/dyadic_fourier.py`、Tensor計算は
`_backends/torch/profiles/dyadic_fourier.py`に置く。runtimeから研究コードへ依存しない。
`components(spec,sigma)`はL、重み付きcosine係数、そのsigma微分を返し、
`values(spec,displacement,sigma)`は中心移動を適用したraw値を返す。
係数準備はFP64。幾何・正規化・更新は既存Torch profile経路を利用する。
中心でsqrt(squared_distance)の微分が発散しないよう、cosineを偶関数の級数で評価する。

profile-productは座標ごとのraw値を掛け、実際のsiteに対して

\[
D_a=\max\left(\prod_d\sqrt{\sum_{i_d}F_{ad}(i_d)^2},\epsilon\right),\quad
W_a=\frac{\mathrm{amp}_a}{D_a}\prod_dF_{ad}.
\]

floorは全積に一度だけ適用する。粗い格子のalias、切り出された支持、尺度混合の
交差項もsite値からのnormに含まれる。連続積分や対角Fourier normでは置き換えない。
学習値・係数・入力集約はstepをまたいでcacheせず、固定の射影行列だけを共有する。

```python
import torch
from torchcst import (
    BandwidthBounds, CSTLinear, DyadicFourierSpec, chart_presets, presets,
)

profile = DyadicFourierSpec(modes=8, minimum_half_width=.4)
kernel = presets.polar_profile_product(
    profiles=(profile, profile), amplitude_max=1., w_c=.1,
    bounds=BandwidthBounds(minimum=.1, birth=.25, maximum=3.2, upper_floor=.1),
)
layer = CSTLinear(
    chart=chart_presets.regular_grid(((32,), (64,)), spacing=.1),
    atoms=8, kernel=kernel, backend="factored", dtype=torch.float64,
)
x = torch.randn(4, 64, dtype=torch.float64, requires_grad=True)
layer(x).square().mean().backward()
```

既存Polar productではatomごとの幅を共有する既存座標則を維持し、task VJPは
現在のsigmaを固定する。profileを追加してその規則を変更してはいない。
profile単体の参照ではsigmaをdetachせず微分でき、中心・sigmaを各軸で独立に
持つ積についても出力・入力・振幅・中心・sigma勾配を独立oracleで確認する。

## 検証範囲

`tests/test_dyadic_fourier_profile.py`は、192点Gauss積分と別のKKT連立方程式を
独立oracleとしてM=4/8を照合する。尺度移行の両端の2階微分、中心、支持外0、
解析中心・precision微分、粗い格子の離散L2、空支持、floorの両側、stepごとの
可変sigma、公開materialized/factoredのY/dX/全Polar source勾配、
独立な軸別sigmaの3次元積、checkpoint、低精度での有限性も確認する。
凍結済みbottom-up研究の関数・中心／sigma微分とも照合する。

この追加はTorch参照の接続であり、階層集約executorのruntime登録ではない。
以前測定した読み取り量削減は研究用CPU実装の結果。
このTorch参照の速度、CUDA Graph、GPU速度とピークメモリは未測定。
旧Fourier専用Planの3タイル支持を新しいprofileへ流用しない。

再現：`PYTHONPATH=src:. python -m pytest tests/test_dyadic_fourier_profile.py -q`。

2026-10-21のCPU検証は新規28件を含め2152 passed /5651 skipped /18 warnings
（46.59秒）。GPU・DBのskipは実機検証に数えない。Ruff、diff検査とwheel/sdist buildも
通過した。raw logsはignored `output/dyadic-fourier-profile-20261021/`へ保存する。
