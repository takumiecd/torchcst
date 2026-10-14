# 固定幅混合Matérn profileと共有集計

2026-10-14。Fourierの方向は停止し、新しい数学的kernel revision6として定義する。
各atomは固有のlive sigmaを持ち、共通の固定幅profileの隣接2点を混ぜる。
profile productのPolar幅更新・task sigma stop-gradientは既存契約を保つ。

## 数学的定義

周期D、幅基準sigma_j、減衰長b_j=sigma_j²。
直線上のf_b(r)=(1+r/b)exp(-r/b)を周期画像の和にし、中心の値で割ってpeak1とする。
最短距離への単純代入とは違う定義で、反対側でも滑らか。

```text
q=exp(-D/b), u=1-q
F_b(d)=exp(-d/b)[(1+d/b)/u+(D/b)q/u²]
G_b(d)=[F_b(d mod D)+F_b(D-d mod D)]/peak_b
peak_b=(1+q)/u+2(D/b)q/u²
j=interval(sigma), t=log(sigma/sigma_j)/log(sigma_{j+1}/sigma_j)
g_sigma=(1-t)G_bj+t G_b(j+1)
```

knotsの外は端のprofileにclamp。宣言は幅boundsをknotsでcoverすることを要求。
FP32 buffer変換の端点丸めにはmetadata表現誤差のみ許す。
混合は非負、peak1、幅方向に連続。ただし幅基準の点ではintrinsic sigma微分に折れがある。
支持は円周全体で、数学的tile切断はない。tile幅はGPUの実行Recipeだけに現れる。

元のsigma²を保持しているのでsigma0.1の直線半値半径は約0.01678、sigma10は167.83。
spacing0.1では最小幅がsitesより狭く、離散ノルムは中心で変わる。
連続空間のL2や中心非依存ノルムに置き換えない。

## 離散L2

```text
n²=(1-t)²G00+2t(1-t)G01+t²G11
G00=sum_i G_bj(x_i-p)²
G01=sum_i G_bj(x_i-p)G_b(j+1)(x_i-p)
G11=sum_i G_b(j+1)(x_i-p)²
```

一様周期gridではd=delta+n*h。各periodic profileは指数×一次式の2項なので、
各交差積は4方向の指数×二次式の和。静的S0/S1/S2を幅pairごとに保存し、
毎callは中心のdeltaからGram値・中心微分を計算する。
負の指数rateは距離の順を反転し、正の大指数と逆数を作らない。
軸ノルムの積の後でfloorを一度だけ適用する。

狭い山では幅の丸めが正規化中心微分に増幅されるため、Polarからの幅解釈と
Gram/係数の準備はFP64で行う。Torch参照もprofileとノルムをFP64で計算した後に
factorをParameterのdtypeへ戻す。共有状態と保存HはFP32、atom VJPのbatch縮約と
Polar振幅のJacobianはFP64で計算し、返す勾配はParameterのdtypeにする。
Polar原点の保護値tinyは元のParameter dtypeの値を保ち、FP64昇格で変更しない。

## 共有処理と寿命

幅基準ごとに左右の2状態を計算する。1step進む際の距離a=h/b、e=exp(-a)について

```text
z0'=e*z0+X
z1'=e*(z1+a*z0)
```

periodic境界を等比級数で閉じ、Triton associative scanで実行blockを分割する。
各atomは自分の中心に隣接する左右siteから、2幅だけを読み取る。
出力側は同じ線形集計の随伴としてatom寄与を散布し、逆方向のscanで復元する。
全site×atomのfactorをshared routeで生成しない。
Hと中心微分をbackward用に保存し、入力のmoment作業領域は縮約後に解放する。
immutable幅/geometry tablesだけをcacheし、live幅・中心・係数はcallごとに読み直す。

研究Algorithm `research_cuda_matern_mixture_triton/v1` はbenchmark registryのみ。
Torch参照は公開preset `polar_matern_mixture_product` で使用できる。
GPUの対応範囲はFP32 IEEE、8/16knots、2つの一様周期軸、64/128sitesの実行block。
数学的なtile幅制限ではなく、初回の実装対応範囲。
座標buffer（origin/periods/RegularGrid spacing）はFP64を要求する。
FP32のchartはsiteごとに座標を丸めるため、等間隔のFP64 scanと異なる離散ノルムに
なりうる。Torch参照は元のchartのsite座標を保ち、研究CUDAはこの違いを黙って許容しない。
研究routeを使う場合は設定境界で`model.chart.double()`を行う。

## 初回の固定比較

N1024、A=int(.05*1024²)=52428、B32、D102.4、spacing0.1、sigma0.1..10、8/16幅。
独立native CPU FP64 oracleでY/dX/all-dPを全sites/全atoms、初期と24更新後に検査。
max_abs/relativeL2 gateは4e-4。norm floor、retained saves、live幅、Graph/public optimizerも検査。
完全stepはprepare/L2/forward/MSE/backward/fusedAdamW/Polar/periodic updateを含む。
未計装の隔離workerで中央値とcapture/replay allocated/reservedを記録。
L4で実測済み。共有実装は全atomの正しさを保って直接実装を大幅に上回るが、
このdensity条件ではdenseより遅い。公開CUDA dispatcherには採用しない。

```python
from torchcst import MaternMixtureSpec, BandwidthBounds, presets
profile=MaternMixtureSpec()
kernel=presets.polar_matern_mixture_product(
    profiles=(profile,profile), amplitude_max=1., w_c=.1,
    bounds=BandwidthBounds(minimum=.1,birth=.1,maximum=10.,upper_floor=.1))
```

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/matern-mixture-1024-a52428-b32-k16.json \
  --plans matern-shared matern-direct dense --output output/matern/cohort.json
```

## 2026-10-14の検証結果

source `a7f23ce655d10525bb3ae49d0a2cc106dc33a7d6`、NVIDIA L4、
Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0、IEEE FP32、座標FP64、実行block64。
CPU1,900件、L4上82件の回帰検査が合格した。後者は実GPU66件とCPU原点検査16件。
8/16幅それぞれ2回の隔離process cohortを逆順で実行し、各workerで21時間sample、
実際のoptimizer更新24回を確認した。21 samplesを21独立runと数えない。
初期と更新後のY/dX/全52,428 atomsのdPは独立FP64 oracleの4e-4 gateを通過。
共有実装の最大絶対誤差はY8.51e-6、dX6.66e-6、all-dP1.044e-4。
全atomのlive幅が更新された。原点・floor・retained saves・Graphも検証した。

| 完全step | 中央値ms（2runsの範囲） | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: |
| 共有8幅 | 2.7968–2.7988 | 47.17 | 82 |
| 共有16幅 | 2.8771–2.8777 | 55.30 | 82 |
| 直接8/16幅 | 232.42–232.94 | 889.66 | 922 |
| dense | 0.08447–0.08475 | 33.00 | 86 |

allocated/reservedはwarmup/capture/replay込み。独立CPU oracleと診断を除く。
GPU process全体のVRAMは未測定。denseは同じshape/batch/精度/optimizerの工学的controlで、
同一parameterizationの精度比較ではない。runnerの`qualifies=true`は直接実装との比較であり、
denseへの採用を意味しない。共有は直接比約81–83倍速くallocatedを約94–95%減らしたが、
dense比では約33–34倍遅く、allocatedも約1.43–1.68倍だった。

別Graphでの診断では、固定初期forwardの幅・L2・係数準備が両bankとも約1.221ms、
入力共有scanは8幅0.0266ms/16幅0.0502ms、atom queryは約0.042–0.043ms、
出力scatter/随伴scanは約0.146–0.173msだった。更新後の別copyでは
forward/loss約1.44–1.48ms、backward約1.19–1.23ms、optimizer約0.171–0.172ms。
これらの計装値を主測定へ足し合わせたり差し引いたりしない。
共有scan自体は軽く、次の最適化候補はatomごとの幅/L2/係数準備とbackward側の集計。
profile定義と正しさを先に確立し、速度最適化は次の段階とする。

再現とraw evidenceの保全先は
[研究記録](research-history/cuda-linear/20261014-matern-mixture.md)に記載する。
