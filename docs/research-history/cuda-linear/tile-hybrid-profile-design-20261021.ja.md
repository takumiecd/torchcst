# 多項式 / Fourier混合kernelとタイル保存・端タイルの一時集約

2026-10-21。CPUの関数設計prototype。公開KernelSpec、Torch backend、CUDA実装への登録はない。
中心muとsigmaはatomごとに自由。旧GPU最適化の再開ではない。
先のcosine座標2pieceとは別の族で、**純粋な距離多項式**を小sigma側に使う。

## 数式の契約

\[
r=x-\mu,\quad R=\sqrt{11}\sigma,\quad
P(r,\sigma)=\begin{cases}(1-(r/R)^2)^4,&|r|<R,\\0,&\text{otherwise}.\end{cases}
\]

Pの連続L1標準偏差はsigma。中心peak1、非負、支持端へC3で接続する。
Fourier近似後の実効幅は測って報告する。sigmaという名前だけで全域の幅一致を主張しない。

\[
F_{L,M}(r,\sigma)=a_0+\sum_{k=1}^M a_k\cos(k\pi r/L),\quad |r|<L,
\qquad F=0\ (|r|\ge L).
\]

基本周期2Lで、周期複製はしない。追加の空間窓は使わない。
係数は、Pへの連続二乗誤差を最小にする制約付き射影で定義する。
端の偶数階微分0をnu個、中心値1を条件とする。

\[
\min_a\int_{-L}^L(F_{L,M}-P)^2\,dr,\quad
\sum_{k=0}^M(-1)^k k^{2j}a_k=0\ (j=0,\ldots,\nu-1),\quad
\sum_{k=0}^M a_k=1.
\]

j=0のk=0項は1。sigma非依存の行列Aに条件をまとめ、
W=diag(2,1,...,1)、b=(0,...,0,1)として、通常の打切りcosine係数a*から

\[
a=a^*+W^{-1}A^T(AW^{-1}A^T)^{-1}(b-Aa^*)
\]

を得る。連続L2の重みWを使う点が旧係数Euclidean射影と異なる。
W、Aは小さい固定行列。a*とsigma微分は母多項式の解析的積分から求める。
端条件はkernelの定義に含める。制約から非負性・単峰性は保証されないため検査する。
M4でnu2/3、M8でnu2/3/4を調べる。基本候補はM4 nu2(C3)、M8 nu3(C5)。

### sigmaの移行とdyadic L

T=spacing*tile_points、L0=h*T、hは2/4/8/16候補。
S(t)=6t^5-15t^4+10t^3を[0,1]で用い、外側は0/1に固定する。

\[
g(R;b)=S\left(\frac{R-(1-\eta)b}{2\eta b}\right),\quad
\eta=.12,\quad \alpha=g(R;.65L_0).
\]

R/L0の.572〜.728で多項式からFourierへ移る。
L_j=2^j L0、b_j=.85L_j、beta_j=g(R;b_j)とし、
Rが[(1-eta)b_j,(1+eta)b_j]内では

\[
Q=(1-\alpha)P+\alpha[(1-\beta_j)F_{L_j,M}+\beta_jF_{2L_j,M}].
\]

尺度移行区間外は対応する単一Fを使う。Rが大きくなるほどLを倍にする同じdyadic集合であり、
sigmaが小さくなる場合は半減となる。移行時だけ2つのLを共存させる。
各成分は自身の支持外0で、全体は最大active Lの外で0。
eta=.12ではnative移行と最初の尺度移行は重ならず、同時に最大2成分。

重みのsigma一次・二次微分は移行区間の端で0。
PとFの厳密一致を課さず、**新しいQ自体をsigmaに関してC2**にする。
Q_sigmaにはalpha_sigma*(F-P)、尺度移行ではbeta_sigma*(F_2L-F_L)を必ず含める。
狭い移行でこれらが大きくなる点も母形状との勾配差として評価する。
mu方向は重みが一定で、支持端を含めてC3以上。

### sampled L2とprofile_product

E=sum_i Q_i²、Em=sum_i Q_i Q_mu,i、Es=sum_i Q_i Q_sigma,iとし、
K=Q/max(sqrt(E),1e-12)。floorの有効/無効を含めた同じ微分を全方式で使う。
混合前の各成分を別々に正規化してから足す方式にはしない。
Fourier対の交差項は係数convolutionと格子上の等比級数geometryで計算。
PとFの交差項はnativeが残る限定領域のsite和で求める。

2次元の直積格子ではQx*Qyとし、L2²=Ex*Ey。
中心・sigmaの勾配も各軸の積で表せる。独立Torchで2軸の値・dX・各軸勾配を検査する。
ただし任意のX[i,j]は分離できないので、共有bankは二重momentが必要。
周波数数Fx,Fyなら容量O(batch*Ntiles*Fx*Fy)であり、O(Fx+Fy)とはしない。

## 実行方式: coarse tile + streamed boundary tile

常設する入力依存量は各tileのFourier moment（この参照ではそのtile方向prefix）。
atomごとのquery区間は、完全に含むtileと左右高々2つの部分tileへ分割する。
完全tileは常設bankから読み、端のqueryを**周波数・端tile番号**でまとめる。
そのgroupのtileだけsite方向prefixを一時構築して全queryに配り、破棄する。

- 支持端の裾を捨てない。muがtile境界を越えても元のQの値・勾配を保つ。
- 端tileではX*cos/sinの入力集約を構築する。各atomとのkernel評価をsiteごとに実行しない。
- 大きい支持全体を再計算する必要はないが、境界処理の計算量は0ではない。
- Xが変わればbankと端groupの値は再構築する。mu/sigmaの変化でもgroupmetadataを更新する。
- atomのParameter配列を並べ替えず、queryのgroupmetadataだけ作る。

batch B、Nsite、tile点数n、live周波数F、atom A、edge group Gについて、
常設bankはO(B*F*N/n)、一時prefixはO(B*n)、querymetadataはO(A*M)。
入力momentの構築は全tileでO(B*N*F)、端集約はO(B*n*G)。
Gは独立query数以下。同じ端tile/周波数を共有するatomが多いほど端構築も減る。
L2のgeometry、係数構築、grouping、全体のHとmu/sigma pullbackをCPU測定に含める。
Python object込みメモリは別にtracemallocで測る。配列payloadの和をpeakと呼ばない。

2次元での端は部分長方形と隅を扱う必要があり、1軸のGやscratch式をそのまま適用しない。
今回のstream executorの性能測定は1軸のみ。2軸profileの数式/勾配確認とは範囲が異なる。

## CPU検査protocol

1. 独立Torch quadrature/autogradで、合成H出力、入力X・振幅・全中心・全sigma勾配を確認。
2. native移行とdyadic移行の全端、中心のtile越境、L2、有限差分を検査。
3. M4/8、nu、L0/Tを変え、形・勾配の母形状との差、負の裾、幅を測る。
4. 部分tileを丸ごと捨てる近似を別診断とし、exact executorの誤差と混ぜない。
5. N4096、B8、A16/128、tile32/64/128、L0/T4/8、3動的step。
   sigmaはatomごとに.1〜40のlog分布、stepごとにX・mu・sigmaを変更。
   全直接、全siteprefix、coarse+streamを同じ入力で比較する。

時間は各step3回中央値。CPUの軸別Hとpullbackの参照であり、
CSTのY/dX全体・optimizer・GPU完全stepの測定ではない。Dense比較はこの段階では行わない。
メモリは入力fixtureを作った後のcall内tracemalloc peakで、RSS/allocator peakではない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_tile_hybrid_profile_cpu \
  --output output/tile-hybrid-profile-cpu-20261021
```

CPU結果は検証済みcheckpointで追記する。
