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

## CPU結果

測定source `657345929815c85ad7cba7ec45d13c8e110af7d4`、Apple M4 Pro、
NumPy/Python、BLAS thread1。原本・図・source snapshot・hash照合はignored
`output/scale-matched-cosine-cpu-20261021/`に保存。
Ruff pass、新規18test pass、CPU suite **2098 passed / 5651 skipped / 18 warnings (41.76s)**。
skipをGPU確認と数えない。

| 指標 | M4 | M8 |
| --- | ---: | ---: |
| rho | .1236925020 | .09785252465 |
| 校正後の連続幅の最大相対誤差 | 7.58e-10 | 8.38e-10 |
| 幅微分の最大相対誤差 | 7.44e-8 | 8.25e-8 |
| octave内の標準化形状変化・最大relL2 | **20.06%** | **18.48%** |
| 3sigma外の振幅 / 中心peak・最大 | 2.683% | 2.420% |
| 3sigma外のL2エネルギー割合・最大 | .02028% | .02009% |
| 実支持半径 / sigma・最大 | 5.392 | 6.817 |
| spacing .1、sigma .1でのsampled幅誤差・最大 | **1.718%** | **.07892%** |
| spacing .1、sigma .3でのsampled幅誤差・最大 | .04192% | .0006842% |

形状変化は中心peak1の`q(sigma*u)`をt=0の形と比較。L2正規化済み振幅の誤差ではない。
非負・片側単調は全129phaseで違反0。連続幅の表は513phase、sampled幅は中心17phase。
sigma10まで有限格子上で検査しただけであり、10が数式上の上限ではない。
大きいsigmaは新しい尺度とprefix周波数を必要とする。

Lと2Lを**同じ切替sigma**で強制評価した場合、正規化値・mu微分・sigma微分は
検査した全点で差0。左右にsigma*1e-7ずつずらした通常評価では、
値relL2の変化<=1.69e-7、微分の変化<=9.21e-7で、有限のsigma変化に追従している。

42動的状態で、直接式対Fourier式の最大絶対差は値5.02e-14、mu微分7.38e-14、
sigma微分2.93e-12。共有Hとそのpullback対直接は最大1.53e-9。
有限差分relL2はmu2.17e-10、sigma8.10e-10。
L2geometryの相殺により、Eの最大差1.27e-9、Es=sum(q*q_sigma)の差2.57e-9が残る。
FP64の有限サンプル検査であり、FP32、全CSTのY/dX/全atom勾配のgateではない。

### 構築・正規化を含む軸別CPU時間と保存

各step3回の中央値を取り、さらに動的3stepの中央値を下表に示す。単位ms。
prefix容量は3step中の最大、complex128配列のpayloadでありCPU peak/RSSではない。

| M | tile点数 | atom | 全直接 | 直接/共有の混在 | prefix MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 16 | 1.038 | 9.578 | 1.627 |
| 4 | 32 | 128 | 8.615 | 74.254 | 1.627 |
| 4 | 64 | 16 | 1.050 | 6.318 | 1.376 |
| 4 | 64 | 128 | 8.747 | 60.024 | 1.376 |
| 4 | 128 | 16 | 1.041 | 5.464 | 1.126 |
| 4 | 128 | 128 | 8.390 | 38.486 | 1.126 |
| 8 | 32 | 16 | 1.100 | 16.579 | 3.128 |
| 8 | 32 | 128 | 9.349 | 132.376 | 3.128 |
| 8 | 64 | 16 | 1.180 | 11.011 | 2.127 |
| 8 | 64 | 128 | 9.343 | 103.520 | 2.628 |
| 8 | 128 | 16 | 1.173 | 9.111 | 2.127 |
| 8 | 128 | 128 | 9.278 | 64.569 | 2.127 |

原本fixture hashでM/tile間のX・mu・sigmaを一致させた。
atom128では周波数prefixの構築本数を、共有atomごとに独立構築する場合から
95.7〜97.3%削減した。それでもPythonの係数・geometry・L2処理をatomごとに行うため、
この参照では共有版が速くなる損益分岐は観測できなかった。
tileを広げると共有に回るatomが減るheuristicの影響もある。
tile32²等の2次元tile集約を実装して測った値ではない。

prefix保存は周波数本数Fに対し16*batch*(N+1)*F bytes。
両経路の返却H/Hmu/Hsigmaは3*batch*atom*8 bytes（atom128で24KiB）、
共通Xは64KiB。直接参照は3*Nのprofile配列（24KiB）等を一時生成する。
校正tableはMごとに520bytesで、atom/sigmaごとのtable保存は不要。
配列一時値やPythonオブジェクトを含むpeakメモリをこの算式で代用しない。

計算量は共有前処理O(batch*N*F)、atomごとの縮約O(batch*M)、
L2は小さい係数畳み込みO(M²)。直接の数学的な支持限定評価はO(batch*S)にできるが、
今回の直接参照は長さNの配列と積を使うので、その部分はO(batch*N)。
このCPU時間をGPU dispatcherの採用根拠やDenseとの速度比較に使わない。

## 判断

**切替点での形・勾配の厳密一致とsigma校正は成立したが、提案全体は未採用。**
元の正係数cosine多項式はdyadic幅を覆えず、緩和した2piece方式も
octave内の形状変化18〜20%とサイトprefix容量を残した。
単一Fourierによるtile全体の集約だけで、厳密な支持半減・全区間一致も同時に満たす、
という条件は成立していない。次の設計では、形状変化の許容値か支持端の微小裾を
明示してから数学的な自由度を決める必要がある。この候補のGPU最適化は開始しない。

その後、ユーザーがsigma移行区間での多項式/Fourier混合を許容したため、
[純粋多項式とFourierの混合・端tileのstream集約](tile-hybrid-profile-design-20261021.ja.md)を
別候補として検討した。本ノートの18〜20%の形状変化を、その新候補の結果へ流用しない。

![shape and seam](../../../output/scale-matched-cosine-cpu-20261021/scale-matched-shape.png)
