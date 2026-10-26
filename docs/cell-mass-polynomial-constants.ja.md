# Fourier係数のmode定数を先に用意する

`coefficient_method="polynomial_table"`は、従来の`polynomial`と同じ正規化済み
ChebyshevSquareの係数を、少ない割り算で評価する方式。
DFT・scalar polynomialも比較用に残す。既定値は従来のpolynomial。

## atomに依存するものと、mode数だけのもの

\(x=\beta^2\)、\(t=x-1\)として、raw Fourier係数の式には

\[
\frac{(m-k-j)(m+k+j)}{j(j+1)},\qquad \frac{m}{m+k}
\]

という、mode数mとharmonic k、和の番号jだけで決まる比率が現れる。
これをatomごとにdoubleで割り算せず、m=2,4,…,16の小さな定数表から読む。
表はコンパイル時に生成し、CUDA constant memoryで9864bytes。
atomやsigmaの値を保存する係数表ではない。

atom側ではbeta/sigmaから次の3量を一度計算して再利用する。

\[
\partial_\sigma\log x=\frac{2\beta_\sigma}{\beta},\qquad
s=\frac{2}{P c_0},\qquad
\partial_\sigma\log c_0=\frac{(c_0)_\sigma}{c_0}.
\]

harmonicのraw係数をr_kとすると、正規化した値と幅微分は

\[
\widehat r_k=s r_k,\qquad
\partial_\sigma\widehat r_k=s\left((r_k)_\sigma-r_k\frac{(c_0)_\sigma}{c_0}\right).
\]

raw係数・その微分・位相は引き続きcompute中に生成する。
全atomの係数やJacobianをglobalへ展開しない。内部精度はdoubleを維持する。

## 同じGPU・同じsourceでの実測

source `839e3459`、L4 / Torch2.11.0+cu130 / CUDA13.0。
N1024/A52428/B4、FP32、double Z、owner C、G/Parameter勾配融合、
幅256/split128/64thread。固定X/dYのforward/backward/fused AdamW。
2fresh process、逆順、warmup10/同期wall30標本。

| 方式 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| scalar polynomial | 3.318 / 3.319 | 5.261 | 6 |
| polynomial_table | 2.643 / 2.651 | 5.261 | 6 |
| table・128thread | 2.694 / 2.698 | 5.261 | 6 |
| table・G別生成・64thread | 3.004 / 3.004 | 5.530 | 8 |
| dense通常 | 0.467 / 0.438 | 32.313 | 42 |
| dense workspaceなし | 0.638 / 0.626 | 16.063 | 22 |

約20%短縮。時間は通常denseの5.66/6.06倍で、約2倍の目標には未達。
表のallocatorピークにはCUDA module内の定数表9864bytesを含めない。
その固定容量も別に記録し、全atomに比例する保存量と区別する。
reservedもworkspaceなしdense以下。全GPU processの消費量を測った値ではない。

別測定の64threadではH単独0.438→0.315ms、G/Parameter勾配融合1.439→1.120ms、
owner Y0.523→0.407ms、owner dX0.469→0.360ms。
全体wall時間はそれらの和で代用しない。

128/A819/B4・微分保存・幅16/split8/16threadも同じsourceで確認した。
scalarは0.857/0.855ms、tableは0.728/0.738ms、通常denseは0.442/0.439ms。
tableの時間比は1.65/1.68。allocated200KiB / reserved2MiBで、
workspaceなしdense264.5KiB / 2MiBを下回る（reservedは同じ）。

## 検証と証拠

CPU2006 passed / 3567 skipped / 既存警告18件（42.56秒）。
CUDA・数学819 passed、failed/skippedは0。
全mode2..16、FP32/64、DFT/両polynomial、両Z方式、owner C有無、
保存・再計算・G融合、0/端数batchとatom、周期端、retained backwardを含む。
全52,428atomのY/dX/全勾配と、GPU fused AdamW更新・moments・stepもCPU oracleへ照合した。

job `l4job-62c8a2e84de143b9bb31d728703ccc9a`:

- source SHA256: `60dfb9d14cb416dc9bc10cbd681c0b749dd3599ad1785ffb51562ea01ab05afe`
- result SHA256: `e4fabeae1301ee06748975a8818ddc05e1bfc95ccca503c69be6bda85e896211`
- driver SHA256: `7f4d2fd30965c0fb3e3f12e0d4752e6a73dba4d46c1638919a2d78d37dfc10c5`
- binary SHA256: `743bb113278825518769d2e631b1e355cd0632b05dc2f250b4abe3a6e764fd87`

144 manifest files、321 runtime sources、driver、128 helper、binaryを照合。
rawはignored `output/cell-mass-1024-20261027/table/`。
再実行はsource archiveの`__pool_driver__.py`へsrc/rootのPYTHONPATHと空directoryの
`CST_JOB_OUTPUT`を渡す。公開dispatcher・Graphへの接続は別の作業。
