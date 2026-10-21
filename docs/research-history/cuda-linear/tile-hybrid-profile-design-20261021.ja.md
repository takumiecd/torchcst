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
測定ではConfigへ`overlap=.12`を明示する（基底Configから継承する既定値は.08）。

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



端tileを厳密処理できるため、余白のためだけにL0を広くする必要はない。
最初のL0/T4/8測定を保存した後、追加のCPU測定でL0/T1/2も比較する。
同じ入力seed・stepを維持し、既存条件の反復測定と混同しない。
追加測定は`--benchmark-only --minimum-L-tiles 1 2`で実行する。

## 検証結果と判断

最初のsource `f3da519f6857e8130dbf2931e0b49dd1f8e557d8`、追加のsource
`7b139a9e`。両sourceのkernel・helper・testは同じbyte。
追加sourceはrunnerのL0/T選択とbenchmark-onlyのCLIを追加しただけ。
原本はignored `output/tile-hybrid-profile-cpu-20261021/`と
`output/tile-hybrid-profile-small-period-cpu-20261021/`。
各proof.jsonにsource snapshot/hashとartifact hashを保存した。
全M/tile/L0/Tのfixture hashをatom数・stepごとに照合し、一致を確認。
Apple M4 Pro、NumPy、BLAS/OMP/Accelerate各1threadで逐次測定した。

新規11test pass。独立Torch quadrature/autogradで、軸別の合成出力・入力勾配・
全atomの振幅/中心/sigma勾配、および2軸profile productの値・入力/中心/sigma勾配を確認。
照合toleranceはabs2e-10、rel2e-9（case別により厳しい検査もある）。
CPU suiteは**2109 passed / 5651 skipped / 18 warnings (43.70s)**。Ruffとdiff checkもpass。
完全CSTの2次元共有executorのgateではない。

### 形とsigma

形状1160条件、端tile省略を中心5phaseで別検査した。
以下は中心・sigmaを固定した**母多項式との比較**であり、
新しいQ自身の実装の微分誤差ではない。shapeはsampled L2正規化後のrelL2。

| 指標の最大値 | M4 nu2 | M8 nu2 | M8 nu3 |
| --- | ---: | ---: | ---: |
| 形状差 | 5.2438% | 0.20508% | 0.2393% |
| 中心勾配の差 | 13.748% | 1.2142% | 1.3656% |
| sigma勾配の差 | 61.428% | 3.2651% | 4.2136% |
| 負の振れ / 中心peak | 3.7038% | 0.16974% | 0.19915% |
| 母多項式支持外の振幅 / peak | 3.4753% | 0.13636% | 0.15918% |
| 母多項式支持外のL2 energy割合 | 0.15097% | 0.00016775% | 0.00018615% |

M8 nu3のenergy RMS幅 / 母形状の幅は.999387〜1.000144。
正の部分だけを取ったL1標準偏差 / sigmaは1〜1.009174。
負の裾があるQを非負の確率分布として扱っていない。
M4 nu2はそれぞれ最大1.06099、1.37437で、弱い遠方の裾も幅を変える。
sigmaを連続学習する自由は残るが、混合後まで厳密な標準偏差とする校正は行っていない。
spacing=.1、sigma=.1の小sigma域は両Mとも同じPで、中心17phaseでsampled幅の誤差は最大.04094%。

端条件を増やしたM8 nu4はshape最大.27964%、sigma勾配差6.2142%。
M4 nu3はshape10.524%、sigma勾配差116.68%。端を平坦化するだけでは形が改善しない。
nu2は母形状への近さ、nu3は端の平坦さを優先する選択肢で、どちらも厳密な非負性は未保証。
単純なmax(Q,0)はFourier集約を保たないため、実装に追加していない。

### 接続と実装の精度

sigmaを移行端の左右へ相対1e-8ずらした差は、値/勾配で最大1.61e-7。
中心をtile境界の左右へ1e-10*Tずらした差も0へ近づく。元のQはC2接続で、
stream方式は同じQを評価する。測定領域では正規化floorは無効。
短い重なりに移行誤差を隠したhard切替ではない。
H・Hmu・Hsigmaのstream対直接の最大絶対差は、全144動的benchmark状態で7.67e-14以下。
検査用のE/Em/Esも直接和との差は8.53e-14以下。

一方、端tileを捨てる近似は別のkernelになる。
L0/T8のM8 nu3でも、中心tile境界の左右でshape差9.41e-7、sigma勾配差2.76e-4が残った。
小さい誤差でも厳密な滑らかさと混同しない。**推奨するexact方式は端tileを捨てない。**

### CPU時間とメモリ

下表はA128、B8、N4096。時間は各step3回中央値、さらに動的3stepの中央値（ms）。
M4はnu2、M8はnu3。元のsourceと追加sourceは同じ明示設定eta=.12で測定。
Python参照の全係数・groupmetadata・入力bank構築・L2とH/pullbackを含む。
GPUまたは完全CST step時間ではない。CPU共有版が直接版を上回る条件は観測しなかった。

| M | tile点数 | L0/T | 全直接 ms | 全site prefix ms | coarse+stream ms | 全site prefix MiB | 常設tile bank MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 1 | 33.042 | 106.477 | 117.672 | 16.504 | 0.520 |
| 4 | 32 | 2 | 31.424 | 92.876 | 103.681 | 14.504 | 0.457 |
| 4 | 32 | 8 | 28.182 | 67.426 | 73.326 | 10.503 | 0.331 |
| 4 | 64 | 1 | 31.405 | 93.443 | 101.733 | 14.504 | 0.230 |
| 4 | 128 | 1 | 29.383 | 78.925 | 85.346 | 12.503 | 0.101 |
| 4 | 128 | 8 | 17.623 | 26.581 | 27.088 | 6.502 | 0.052 |
| 8 | 32 | 1 | 44.951 | 183.093 | 205.946 | 32.508 | 1.024 |
| 8 | 32 | 2 | 43.724 | 159.176 | 178.494 | 28.507 | 0.898 |
| 8 | 32 | 8 | 37.670 | 109.819 | 124.588 | 20.505 | 0.646 |
| 8 | 64 | 1 | 43.176 | 161.330 | 174.945 | 28.507 | 0.452 |
| 8 | 128 | 1 | 40.618 | 135.389 | 146.578 | 24.506 | 0.197 |
| 8 | 128 | 8 | 23.769 | 41.754 | 43.097 | 12.503 | 0.101 |

代表例M8/tile32/L0=T/A128のtracemalloc追加peakは、直接2.916MiB、
全siteprefix33.112MiB、coarse+stream2.105MiB。
入力fixtureを除くcall内のPython/NumPy追跡値であり、RSS・GPU peak・モデル全体ではない。
常設bank1.024MiBと、一時buffer・querymetadata・係数・結果・計算中の配列が別に必要。
境界scratchのstatsは選択した配列の容量モデルであり、実測peakを代用しない。

M8/tile32/L0=Tでは、共有atom数を16から128へ増やすと、端group構築の削減率は
独立queryごとの構築に比べ約33%から76%になった。tile128/L0=T/A128では約90%。
構築回数の削減率と速度向上率は別であり、Pythonのatom別L2/geometryとgroup処理は残る。
CPU原型だけからGPUの律速箇所や最適なnative切替を決めない。

### 実装に進める条件

**M8は、近い山形を保つ混合kernelの候補として成立。M4は現設定で推奨しない。**
小sigmaは純粋多項式、大sigmaは有限Fourier、移行では両方を使い、
追加窓を掛けずに中心・sigmaの自由と厳密な支持・微分を維持できた。
タイルbankと端tileのstreamにより、global siteprefixの常設保存を避けられた。
ただし負の小さい裾を許容する設計であり、非負性を必須にする場合は未完成。
CPU速度の利益はまだ出ておらず、GPU高速化・Dense目標の達成を示す結果ではない。

sigma_min=.1、spacing=.1の小sigmaは6〜7siteのnative評価で済む。
32点tileでL0=Tならnative/Fourier移行はsigma約.552〜.702、L0=2Tなら約1.104〜1.405。
L0を広げるとnativeで処理する支持も広がる。最適値は構築・共有atom数込みの実測で選ぶ。
sigma40まで測定したことは、40が数学的上限という意味ではない。
上限は最大Lのbank容量と、許容する入力域切断の条件から別途決める。

2次元ではprofile/L2の軸分解ができても、bank容量の周波数積は残る。
1尺度complex128 bankとfloat64入力のpayload比は概ね2*(2M+1)^2/n^2で、
M8/n32では.564、M8/n128では.0353。これは2D実測ではなく、1尺度だけの容量モデル。
多尺度の全Fx*Fyを持てば増える。1軸で省メモリ化した結果を2D全体へ外挿しない。
次に必要なのは、このM8候補の2D共有executorでの構築・L2・backward込みの評価である。

![hybrid function and errors](../../../output/tile-hybrid-profile-cpu-20261021/hybrid-function.png)
