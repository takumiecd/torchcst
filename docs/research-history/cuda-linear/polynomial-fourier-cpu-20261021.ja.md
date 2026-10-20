# 局所多項式 / Fourier共有の新kernel候補: CPU screen

2026-10-21。旧quadratic B-spline CUDA最適化は一時停止のまま。
この調査は新しい数学的kernel候補の形・連続性・計算量をCPU FP64で比較する。
runtimeのKernelSpec、既定dispatcher、optimizer、GPU実装は変更しない。

## 母形状と幅の意味

\[
P(u)=(1-u^2)^4_+,\quad R=\sqrt{11}\sigma,\quad
p(x;\mu,\sigma)=P((x-\mu)/R).
\]

無限連続領域でL1正規化したpの標準偏差はsigma。
L2正規化後のエネルギー分布p²の幅は別であり、母形状とのenergy RMS比で比較する。
固定格子の切り出し・Fourier混合後にこのsigmaが厳密な標準偏差になるとは仮定しない。
中心とsigmaはatomごとに連続。幅の微分も調査し、既存Polarの幅stop-gradientとは別範囲。

周期Lでの複素級数は正負のharmonic -M..M（M=4/8）、real feature数は1+2M。

\[
c_k(R,L)=\frac RL\int_{-1}^1(1-u^2)^4\cos(2\pi kRu/L)du,\qquad
f_L(x)=\sum_{k=-M}^{M}c_ke^{2\pi ik(x-\mu)/L}.
\]

係数は球Besselの閉形式 `I(z)=768 j4(z)/z^4` と
`z I'(z)=-768 j5(z)/z^3` から計算する。z<4は20項の安定級数、
z>=4はsin/cosの有理式。係数とsigma微分はO(M)、小zの巨大な差分は避ける。
独立Gauss-Legendre128点と適応積分で照合する。

## タイル、周期、切替

spacing=.1、1軸tile幅は32/64/128点。Tを物理tile幅とする。
`L_j=L0*2^j`。L0=4T/rho=.22、L0=8T/rho=.34を対照にする。
各周期の有効集合は `K=L/T-1` 個の隣接する丸ごとのtile。
centerの所属tileを中心とする奇数個の集合であり、native支持を囲む余白を持たせる。
Fourier側の境界サイトをnativeで直接計算する補正はしない。

native/FourierはR/L0=.16..22、周期間はR=rho*Ljの±8%で、
quintic smoothstep `s(t)=6t^5-15t^4+10t^3` により混ぜる。
その重みのsigma微分も含める。hard switchは負の対照として残す。
早期切替対照はR/L0=.05..075（L0=8Tなら支持直径が約.8T..1.2T）。
大きいtileでも「1tile程度で即Fourier」が成立するかを別に調べる。

中心の所属tileによる不連続を検出するため、hard tile集合と、隣接2集合を
`s(frac(mu/T))` で混ぜる `tile_blend=True` を分ける。
後者のmu微分は位相微分だけでなく混合比の微分も含む。
これはnew kernelのスカラー係数としての定義で、既存profileに追加窓を掛けた同義計算ではない。
最大2周期×2集合を参照する。重複tileの入力momentは共有する。
mu/sigmaが滑らかでも空間xのtile端が厳密に滑らかとは限らず、別のmetricで残す。

## sampled L2と共有

入力依存bankは `T[t,k,L,b]=sum_(i in tile t) X[b,i] exp(2pi ik xi/L)`。
入力Xが変われば作り直す。同じXでsigma/centerだけが変わる場合の継続利用と混同しない。
隣接周期の偶数harmonicは同一周波数として共有する。
原型は正負を両方保存したcomplex128。real packingした容量モデルとは区別する。

正規化用のgeometry moment `Z[t,k,L]=sum exp(2pi ik xi/L)` は入力にもatomにも依存しない。
Fourier同士のL2交差項は係数の線形convolutionと、集合の共通tileのgeometry momentで計算する。
単一周期では正の2Mまで、隣接2周期の混合では長周期単位で正の4Mまで必要。
直交性、積分による離散和の代用、tileごとの再正規化は使わない。

native混合区間では `||wp p+sum f||²` のnative/Fourier交差項も含む。
この原型はnative支持のKp点でKp*(2M+1)のmixed和を明示計算し、費用を隠さない。
Fourierのみではnative境界の直接計算はない。全Cartesian productのnormは軸normの積にし、
floorは全積の後に一度だけ適用する検査を行う。

## 検証と計測の範囲

- sigma=.1..10の連続sweepと切替の両側、複数のcenter phase、tile境界両側。
- shapeの母形状との差、normalized mu/sigma勾配の母形状との差、裾の振幅/energy、負の裾、幅。
- candidate自身の解析微分を、独立Torch quadrature/autogradと有限差分で照合。
- sampled L2とその微分、共有H/Hmu/Hsigmaを直接site和と照合。
- 任意の2次元入力に対する軸別productとmaterialized積、全積floorの一致。
- CPU時間は1軸projectionとそのmu/sigma微分、係数・構築・L2込み。Y/dXやoptimizerを含む
  CST完全step、GPU性能、allocator peak、全atom更新を測った値ではない。
- bank payloadは数値arrayのみ。Python辞書、index、temporaries、saved forward、optimizer等を含むpeakではない。

再現（numpy/scipy/matplotlib/torchを含む研究環境）:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_polynomial_fourier_cpu \
  --output output/poly-fourier-cpu-20261021-final
python -m pytest tests/test_polynomial_fourier_cpu_study.py -q
```

結果JSONにはsource commit/2 source hash、環境、全測定sampleを保存する。
raw結果と図はignored outputに保全し、この文書には検証後の要点のみ追記する。
