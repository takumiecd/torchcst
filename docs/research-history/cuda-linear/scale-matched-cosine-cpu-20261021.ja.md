# 切替点で同じ形を持つcosine座標kernelのCPU検討

2026-10-21。数学prototypeであり、公開KernelSpec・GPU実装・dispatcherではない。
以前のGPU最適化は再開しない。中心muとsigmaはatomごとに連続で自由。
以下の候補は**2つの支持区間を持つ区分Fourier**への条件変更を含み、採用済みではない。

## 元の候補の限界

`z=(1+cos(pi*r/L))/2`、`q=z² sum(a_j z^j)`、a_j>=0という候補では、
連続L1標準偏差の範囲がM4で[.211777,.282896]L、M8で[.154314,.282896]L。
どちらも最大/最小が2未満であり、Lを半減した隣の尺度との間に幅の穴が残る。
係数を非負にするこの十分条件の限界であり、全Fourier族の不可能性ではない。

また、全[-L,L]で単一の有限Fourier級数、外側0とする非零kernelを、
支持が半分になった同型kernelに厳密一致させることはできない。
大きい方の級数が開いた余白で0になると、解析性から全区間で0になるため。

## 検査する緩和候補

r=x-mu、M=2p、M4ではp2、M8ではp4とする。

\[
B_{s,c,p}(r)=\begin{cases}
\left(\frac{\cos(\pi r/s)-c}{1-c}\right)^p,&|r|\le s\arccos(c)/\pi,\\
0,&\text{otherwise}.
\end{cases}
\]

周期複製はしない。中心lobeだけを使う。端で値とp-1階までの微分が消える。
M4はC1、M8はC3。cosine座標の多項式であり、距離rの純粋な多項式ではない。

\[
q(r;\sigma)=(1-w(t))B_{L/2,c(t),p}(r)+w(t)B_{L,c(t),p}(r),
\quad t=\log_2\frac{2\sigma}{\rho L},\quad
w=6t^5-15t^4+10t^3.
\]

T=spacing*tile_points、L=T*2^ceil(log2(sigma/(rho*T)))。
小sigmaには負の尺度levelも使う。rhoはc0=.5のBの連続L1標準偏差。
このqの支持は[-L,L]に含まれ、外側は0。内部に2つの支持端を持つ。
これはnativeとFourierの誤差を埋める混合ではなく、両評価で共通のkernel定義。
M4とM8は異なる形の族であり、同じ母関数の近似次数比較ではない。

## sigma校正と接続

v(c)=integral(r² B)/integral(B)をs=1で求めると、

\[
\frac{\operatorname{Var}(q)}{L^2}
=v(c)\frac{1+7w}{4(1+w)}
=\frac{\rho^2 2^{2t}}4.
\]

これをofflineでcについて解く。c(0)=c(1)=c0で、
c_t(0)=c_t(1)=d0=2*log(2)*v(c0)/v'(c0)を満たすよう、
`c=c0+d0*(2t³-3t²+t)+t²(1-t)²*R32(t)`の残差をChebyshev近似する。
近似は幅の校正だけに入り、同じ係数を使う両評価の恒等式には誤差を追加しない。
幅は連続L1分布の標準偏差であり、sampled-gridの標準偏差と区別する。

L側のt=1と2L側のt=0では、どちらもB_{L,c0,p}になる。
さらにc_sigma=d0/(sigma*log(2))も同じなので、値・中心微分・sigma微分が一致する。
w_tを0にするだけでsigma勾配を凍結する設計にはしない。

## 共有評価・L2・容量

各pieceはbinomial展開と`cos^n(theta)=2^-n sum_j binom(n,j) exp(i(n-2j)theta)`
で厳密に有限Fourier展開できる。粗い周期2Lで最大次数2p=M。
直接式と共有式は全sigmaで同じ関数。後付けfitやnative/Fourier切替補正はない。
直接参照は各pieceのcos/sinを別々に評価する。倍角による融合は未実装。

内部支持端を含む任意区間の和が必要なので、今回の共有参照は各サイトで
`P_k(j)=sum_{i<j} X_i exp(i*pi*k*x_i/s)`を保持し、両端の差を取る。
境界サイトとatomの直接評価はしないが、**tile全体の集約だけでは不十分**。
dyadic尺度の偶数周波数は同じprefixを共有。Xが変わるstepではprefixを再構築する。
中心/sigmaが変わっても同じX・尺度ならprefixを再利用できる。係数はatomごとに更新する。

sampled L2のE=sum q²、Em=sum q*q_mu、Es=sum q*q_sigmaは、
piece対の係数畳み込みと、支持の共通部分に対する等比級数geometryで計算する。
2piece一般位置で4対*3種類=12畳み込み。支持端はq,q'が0なので移動境界の項は0。
正規化はq/max(sqrt(E),1e-12)、微分も同じfloor契約。
係数の相殺誤差は測定する。FP32適性は未確認。

profile_productの2次元では軸ごとにこの式を使える。
直積格子上ではsum_{i,j}(q_x(i)q_y(j))²=E_x E_yなのでL2は軸別に分解できる。
一方、任意のX[b,i,j]の共有縮約は2次元の区間momentが必要であり、
この1軸CPU測定を2次元完全stepの速度・メモリと解釈しない。

## 測定protocol

独立Torch自動微分との値・mu/sigma勾配、有限差分、連続幅、支持・裾、L切替を検査。
N1024、batch8、atom16/128、tile32/64/128、spacing .1、sigmaは.1〜10のlog分布。
3stepでX・mu・sigmaを変更する。全直接評価と`sigma<T/4`直接・他共有を比較。
T/4は検査用であり、最適な切替基準ではない。L2・係数・prefix構築込み。
測定対象は軸別Hとそのmu/sigma pullbackで、dX・Y・振幅勾配・optimizerを含む完全CSTではない。
CPU NumPy/Python時間とprefix配列payloadであり、GPU速度・peakメモリではない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_scale_matched_cosine_cpu \
  --output output/scale-matched-cosine-cpu-20261021
```

結果は次の検証済みcheckpointで追記する。形のoctave内変化も失敗条件として残す。
