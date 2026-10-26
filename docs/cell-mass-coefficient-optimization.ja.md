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

## 正しさと実測結果

source commit `5d9f4c36f345127241eb1a6fb7a8a948f0ac619e`、
L4 job `l4job-e077006a4a164b52b937d22cf5c781a9`。
NVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0。
CPU全体は1994 passed / 2820 skipped / 18件の既存JIT警告。
L4環境で対象3ファイルの60件（CUDA 44件・CPU 16件）がpassed、failed/skippedは0。
新しい直接式の全係数とsigma微分、既存セル積分参照、両Z方式、両係数方式、
float32/64、819atom、1024atom上限と範囲外拒否、全atom勾配、保存状態を確認した。
独立参照のFFT/convolution・Gauss積分と既存の誤差基準を保ち、緩和していない。

測定と同じ819atom入力について、Y/dX/全Parameter勾配の最大絶対誤差は次の通り。

| dtype・方式 | Y | dX | 全Parameter勾配 |
| --- | --- | --- | --- |
| FP32・DFT | 2.98e-8 | 5.96e-8 | 5.96e-8 |
| FP32・直接多項式 | 2.98e-8 | 5.96e-8 | 5.96e-8 |
| FP64・DFT | 2.00e-15 | 2.39e-15 | 2.33e-15 |
| FP64・直接多項式 | 1.39e-15 | 1.55e-15 | 1.78e-15 |

時間・メモリは検証processとは別のfresh processで測定した。
FP32・128×128・819atom・batch 4・mode 8、支持半幅は入力24/出力32、spacingは両側1。
sigmaは0.2〜12の線形分布、rho=sigma/(2*spacing)は0.1〜6。
Zはmerge、Z一要素のthread数は4で固定し、変更するのは係数計算方式だけ。
同じバイナリを使用し、二つの独立process/方式でwarmup後に5サンプルずつ測った。

| 方式 | process 1の中央値 | process 2の中央値 | peak allocated | peak reserved |
| --- | --- | --- | --- | --- |
| DFT | 846.732 ms | 846.777 ms | 85,504 byte | 2 MiB |
| 直接多項式 | 68.964 ms | 68.912 ms | 85,504 byte | 2 MiB |

このケースのforward+backwardは約12.3倍。prepare・入力検査を含むwall時間で、
optimizer・Graph・公開Parameterizationは含まない。peak allocatedは両方式とも83.5 KiBで、
基準のX/dY/atomの17 KiBを含む。GPU process全体のVRAMではない。
819atomの主要配列の理論値82.1 KiBと実測83.5 KiBは、確保単位などを含める範囲が異なる。

別のphase診断のGPU event中央値は次の通り。診断の和を全体時間には使わない。

| phase | DFT | 直接多項式 |
| --- | --- | --- |
| prepare | 0.339 ms | 0.348 ms |
| Zx | 0.038 ms | 0.037 ms |
| Zy | 0.035 ms | 0.037 ms |
| H | 37.466 ms | 2.043 ms |
| Y | 376.176 ms | 31.121 ms |
| G | 17.732 ms | 0.979 ms |
| dX | 357.838 ms | 29.473 ms |
| Parameter勾配 | 56.532 ms | 4.420 ms |

同じ演算・入力・割り当てでの切り替えにより、係数DFTの置き換えが改善を生むことを確認した。
現在の既定は直接多項式とし、DFTは同じ数学の比較用として明示的に残す。
今も最も重いのはY/dX。32site×4batchを一つのblockへ割り当てるため、
このshapeではY/dXのgridは各4blockにとどまる。H/GとParameter勾配は各7block。
残る繰り返し計算と並列度の改善は次の比較で扱い、この段階で原因を一つに断定しない。

cuobjdumpのリソース表示では、Y/dXはregister 108/thread・shared 6,176 byte/block・stack 64、
H/Gはregister 92/thread・stack 40、Parameter勾配はregister 168/thread・stack 40。
いずれも表示上LOCALは0で、直接式に係数配列は導入していない。
これはコンパイル時の情報であり、SM占有率やtraffic/stallの実測ではない。
runtimeで方式を切り替えるため、二方式の同じバイナリでのコンパイルリソースは共通である。

source archive SHA256は`73084e371923006604278cb437f4b9641972ed1adffe02a70a0eab7752e008da`、
result archive SHA256は`b77d47745ba31b206a4457e59fbee38a77589e014bdc167e7576fde08ea2d395`。
両archiveのhashと、GPU snapshot内の321 runtime sourceのbyte一致を手元で再確認した。
wheel/sdist buildと321 runtime sourceの照合も通過。
raw evidence・driver・resource dump・package照合はignored `output/cell-mass-opt-20261026/`、
source/result archiveはhost共通poolのjob directoryに保持する。owned L4の停止も確認した。

再実行はCPUで`python -m tools.kernel_dev test --suite cpu`、CUDA/nvcc/Ninjaのある環境で
`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。
この819atom・単一profile/bankの結果を、別shapeやdense完全学習stepの速度勝利へ一般化しない。
