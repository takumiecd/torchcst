# 5% atomでの係数計算の最適化

2026-10-26。[初版の実測](cell-mass-implementation.ja.md)では128atomのforward+backwardが
約206msで、ZよりH/Y/G/dX/Parameter勾配の5段階に時間を使っていた。
本比較は128×128、batch 4、819atom（128²の約5%）へ条件を揃える。
atom上限を1024へ広げ、同じソースから旧DFTと新しい直接多項式の両方を選べるようにする。
Z方式、thread数、profile、支持、normalization、dtypeと入力は両者で固定する。

## 何を変えるか

初版のraw_coefficientは、係数を要求されるたびに2M+1個の点で生密度を評価し、
小さなDFTを計算していた。各評価で三角関数とM段のChebyshev漸化式を実行する。
同じatomの境界セルなどでもこの計算が繰り返される。

係数をglobalへ保存する方針は取らず、同じ係数をスカラー多項式から直接計算する。
新方式もmodeごとにsigmaから評価し、係数配列や係数微分配列を追加しない。
profile・面積正規化・支持境界の処理は変えない。

## 直接計算する式

既存ChebyshevSquareの生密度は、P=2L、theta=2*pi*t/Pとして

\[
 g(t)=\left[\frac{T_M(\beta\cos(\theta/2))}{T_M(\beta)}\right]^2
 =\frac{1+T_{2M}(\beta\cos(\theta/2))}{2T_M(\beta)^2}.
\]

betaは既存のeffective_radiusとsigmaから算出し、beta>=1。
`T_{2M}(beta*cos(theta/2))`の複素Fourier係数をD_kとし、x=beta²、u=x-1と置く。

\[
 D_M=\frac{x^M}{2},
\]

\[
 D_k=\frac{M}{M+k}x^k
 \sum_{j=1}^{M-k}
 \binom{M-k-1}{j-1}\binom{M+k+j-1}{j}u^j,
 \qquad 0\le k<M.
\]

u>=0なので全項が非負。betaの交代符号付き冪展開で大きな値を引き算する方法を避けられる。
和の項をtau_jとすれば、階乗を使わず

\[
 \tau_1=(M+k)u,\qquad
 \tau_{j+1}=\tau_j\,
 \frac{(M-k-j)(M+k+j)}{j(j+1)}u
\]

で進める。x^kも高価な一般powでなく、k回以下の乗算で求める。
生密度の係数は

\[
 c_k=\frac{D_k+\mathbf{1}_{k=0}}{2T_M(\beta)^2},
 \qquad S=P c_0.
\]

sigma微分は同じ漸化式へ伝播する。dx=2*beta*d_beta、du=dxとして、

\[
 d\tau_{j+1}=\frac{(M-k-j)(M+k+j)}{j(j+1)}
             (d\tau_j\,u+\tau_j\,dx).
\]

最後の分母と、セル確率の面積Sの微分も含める。幅の勾配を止めたり、定数扱いしたりしない。
CPUではTorch autograd、CUDAでは明示的なスカラー微分でそれぞれ検証する。

## 実装・検証と比較の入口

- CUDA: kernels.cuのpolynomial_coefficientとdft_coefficient。
- CPUの式検証: torch backendのchebyshev_square_coefficient。
- 独立した既存FFT/convolutionとGauss積分の参照は、置き換えずに残す。
- `physical_linear(..., coefficient_method="polynomial")`と`"dft"`で同じ演算を切り替える。
- 初版と同じ独立Y/dX/全atom勾配ゲートを保ち、819atomとchunk末尾も検証する。

正しさを通過した後にfresh processで時間とpeak allocated/reservedを測る。
段階別診断と未計装のforward+backward全体は別に測り、診断値の和を全体時間にしない。
この比較は低水準physical entryの範囲であり、公開CSTLinear、optimizer、Graph、dense完全stepへの
採用比較まで完了したという意味ではない。
