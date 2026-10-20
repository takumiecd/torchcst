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
結果は検証後に追記する。旧CUDA最適化とGPU実験は再開しない。
