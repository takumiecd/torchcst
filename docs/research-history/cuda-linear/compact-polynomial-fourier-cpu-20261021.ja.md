# 修正した支持契約: [-L,L]、基本周期2L、CPU screen v2

2026-10-21、ユーザーの明示回答により以下へ統一する。

- atom中心からの差rについて支持は[-L,L]、外側は厳密に0。
- Fourier級数の基本周期は2L。LはTに連動したdyadic候補（L0,2L0,...）。
- 中心・sigmaはatomごとに自由。tile集合に合わせてkernelを変えない。
- 小さいsigmaは多項式、広いsigmaはFourier。切替の形・中心/幅勾配とL2を含めて定義する。

前回の[CPU screen v1](polynomial-fourier-cpu-20261021.ja.md)は丸ごとのtile集合で支持を
切る別契約だった。その数値は今回のkernelの評価に流用しない。旧ログとsourceは保全する。

## 多項式 / Fourierを共にC3でzero extensionへ接続

\[
p(r;\sigma)=(1-(r/R)^2)^4_+,\quad R=\sqrt{11}\sigma.
\]

Fourierは `f(r)=a0+sum_(k=1)^M ak cos(k pi r/L)`、M4/8。
母形状の積分係数から、以下を同時に満たす係数へ最小Euclidean変化で射影する。

\[
\sum_k(-1)^ka_k=0,\qquad
\sum_k(-1)^kk^2a_k=0,\qquad
\sum_k a_k=1.
\]

順にf(±L)=0、f''(±L)=0、f(0)=1。
cos級数の奇数階空間微分は端で0なので、支持外0への接続はC3。
中心/幅微分用係数にも同じsigma非依存射影行列を使う。
追加の空間窓、中心を固定するアンカー制約、tile集合間の混合はない。
非負性を保証する制約はまだ入れておらず、負の振れも記録する。

## sigmaとLの切替

T=.1×tile点数、L0=Tまたは2T。native切替はR=.65L0の±8%、
Lから2Lへの切替はR=.85Lの±8%をquintic smoothstepで接続する。
native側の有効時はR<L0、Fourier係数積分時もR<L（子Lで最大.918L）を保つ。
通常は単一L、切替時は2つのdyadic Lを共存させる。支持外0は各成分ごとに適用する。
混合は定義の一部で、重みのsigma微分を含む。L2は混合後のprofileで一度求める。

低次数の純粋多項式と有限Fourierは一般に区間全体で厳密に一致しない。
fの端の条件を合わせることと、sigma切替時に関数全体を合わせることは別である。
hard switchを対照にし、C2の短い切替区間を使うcandidateを検査する。

## 正確な支持とタイル共有に必要な保存量

支持端mu±Lは一般にtileの途中に来る。f(端)=0でも、部分tileの内部すべてが0にはならない。
丸ごとのtile momentだけで部分tileを厳密に切り出せるとは仮定しない。

このCPU共有参照は各周波数・batchについてsiteごとのprefixを持ち、
支持区間の両端2か所をgatherして正確な部分区間和を得る。
これは境界サイトの直接kernel評価を必要としないが、tileだけのbankより大きい。
隣接dyadic Lの偶数harmonicは同じ周波数として共有する。
Xが変わればprefixを作り直す。center/sigmaだけが動く場合は同じXのprefixを共有できる。

L2は係数convolutionと支持区間の幾何的離散和で求め、正規化tableを保存しない。
単一Lで3回、2つのLの混合で12回のconvolution（E,Emu,Esigmaを個別計算）を行う。
native重なりではnative/Fourier交差項も実際のsite和で含める。
queryはO(BM)、入力prefix構築はO(BNF)とO(BNF)容量、Fは全live周波数の数。
正規化は直接convolution O(M²)、native混合区間では追加O(Kp M)。
配列payloadをGPU allocator peak、CST完全step、2次元全体の容量と同一視しない。

再現:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_compact_poly_fourier_cpu \
  --output output/compact-poly-fourier-cpu-20261021
python -m pytest tests/test_compact_poly_fourier_cpu_study.py -q
```

M4/8、spacing=.1、sigma=.1..10、tile32/64/128、L0=T/2Tを比較する。
旧CUDA最適化とGPU実験は再開しない。

## CPU検証済み結果

source `20959f1a74c489379a2f583d18fbe7a0776cb104`。
原本は `output/compact-poly-fourier-cpu-20261021/results.json`、SHA256
`e2b00bc941b7e6d6e5c4e3a7847d0ca6f19f67158d3a83d9a7207548b859fca7`。
新しいcore/runner/testとimportされるv1の係数helperのGit byte一致をproof.jsonへ保存した。
shape7740条件、切替580記録、独立直接和54状態。
macOS arm64、NumPy2.4.6、BLAS/OMP/Accelerateは各1thread。
CPU test終了後に時間測定を行い、同一A/stepのX/mu/sigma hashは全M/tileで一致した。

独立Torch quadrature/autogradと端の0～3階微分・center越境は14test通過。
CPU全suiteは2080 passed /5651 skipped /18既存warnings（42.84s）。
共有norm²/Emu/Esigmaとsite直接和の誤差は5.69e-14/8.06e-16/4.62e-14以下。
共有H/Hmu/Hsigmaと直接値は3.56e-15/5.33e-15/3.34e-14以下。
normalized profile有限差分relL2誤差はmu1.25e-10、sigma1.04e-9以下。
F支持端の値/傾き残差は1.29e-16/2.36e-16以下。数学的なC3条件に対応する浮動小数点残差である。

### 形と切替

主条件は32点tile、L0=T=3.2、sigma=.1..10、複数center phase。
以下の差は母形状pとの差であり、candidate自身の微分計算誤差ではない。

| 最大値 | M4 | M8 |
| --- | ---: | ---: |
| sampled L2正規化後の形状相対L2差 | 5.9268% | .2176% |
| 母形状との中心勾配relL2差 | 15.125% | 1.2802% |
| 母形状とのsigma勾配relL2差 | 105.63% | 4.0163% |
| native支持Rの外、Fourier支持Lの内の裾振幅 / peak | 4.0455% | .1448% |
| 同領域のL2 energy比 | .20484% | .0001766% |
| 負の振れ / peak | 3.3851% | .1698% |
| energy RMS幅 / 母形状の幅 | .999408..1.081480 | .999557..1.000095 |
| 全active成分の支持範囲外 | 0 | 0 |

多項式もFourierも支持端で0へ接続し、tile集合でkernelを変えていない。
muをtile境界±1e-9で評価した差はshape1.33e-8、中心勾配2.21e-8以下で、
有限差分幅での変化であり、v1のtile集合の不連続とは異なる。

sigmaのhard切替はM4で最大8.490%、M8で.1954%のshape差が残る。
端のC3条件だけでは、別の関数であるpとf全体のsigma切替は厳密には一致しない。
quintic重なりでsigmaを相対±1e-7変えたprofile差はM4で2.64e-7、M8で1.57e-7以下。
同条件のsigma勾配差はM4で1.85e-6、M8で5.78e-7以下。
このcandidateは短い接続区間で両表現を使う設計であり、全sigmaで単一表現だけを使う方式ではない。

L0=Tの場合、32点tileのnative/Fourier接続はsigma約.577..677、
native終端の支持直径は1.404T。sigma=.1はnative6～7siteで処理する。
L0=2Tでは接続sigmaも2倍。M8の形状最大差はほぼ同じだが、必要な局所支持は広がる。
Lの段階切替はL/2からLへの変更としても同じ候補集合である。

### 支持端の部分tileは残る

正確な参照はmu±Lのsite区間をprefixで切り出し、境界サイトの直接kernel評価をしない。
一方、prefixを省いて「支持内に完全に含まれるtileだけ」を読む近似は、
L0=Tでは最大約72.4%、L0=2TでもM8で最大4.14%の未正規化profile差となった。
L0=Tの最悪条件はsigma約.6773、mu=3.183。部分tileの中に山の大きな値が含まれるためである。
F(±L)=0という点条件から、部分tile全体が0になるとは言えない。
この差はFourier係数の近似誤差ではなく、不完全な区間集約の誤差。

### 構築・L2込みの時間と容量

N1024、B8、A128、3stepでcenter/sigma/Xを変更。
時間は各step3sampleの中央値を3step平均した1軸CPU Python原型。
Y/dX集約、全atom optimizer、CST完全step、GPU性能の測定ではない。
別契約のv1との時間差を同じ演算の高速化と扱わない。

| 条件 | M4 | M8 |
| --- | ---: | ---: |
| tile32、共有projection＋mu/sigma微分、構築/L2込み | 80.774ms | 135.310ms |
| 同一candidateのsite直接評価 | 17.000ms | 20.564ms |
| L2とその微分だけの別測定 | 65.744ms | 114.775ms |
| tile64、共有版 | 53.504ms | 92.030ms |
| tile128、共有版 | 40.591ms | 67.358ms |

CPU原型は共有版の速度利益を示さず、個別のnorm交差項/geometry/convolution呼び出しが重い。
数学的なO(M²)とPython呼び出しの費用を分け、GPUのボトルネックと断定しない。

tile32、step0でprefixはM4が25周波数・3.128MiB、M8が49周波数・6.131MiB。
同じ周波数を丸ごとのtileにだけ保存する容量モデルは100/196KiBであり、prefixは約32倍。
complex128で正負の周波数を両方持つ原型。real packingすれば数値payloadは概ね半分にできるが未実装。
これはprefix配列だけの容量で、temp、係数、input/output、autograd、optimizer、GPU peakは含まない。
geometry正規化tableは0。M8のtile64/128では周波数が41/33に減り、prefixは5.130/4.129MiB。

tile32でatomごとに別prefixを構築する対照に比べ、構築周波数の重複はM4で96.78%、M8で96.68%削減。
sigma/centerが異なっても入力に対する同一周波数momentは共有できる。
入力Xを更新したstepではprefix自体を再利用せず作り直した。
構築重複の削減率は、全計算量の削減率や高速化率ではない。

## 判断

ユーザーの支持/周期定義で、端を0にしたFourierとC3多項式の接続はCPUで検証できた。
M8は形と幅の保持に有望だが、負の裾とsigma方向の母形状との差は残る。
hardなsigma場合分けで厳密一致するkernelはまだ作っておらず、短い接続区間を採用した候補である。
厳密な支持と部分区間共有をprefixで実現すると保存量が増える。
GPU実装や旧Dense目標の再開は行わず、この契約での形と計算構造の検証結果として保持する。
