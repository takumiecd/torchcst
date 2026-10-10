# 等間隔PeriodicGridのgrouped Wと支持局所H/Gの比較

対象は平坦な2次元Torus、`PeriodicGrid((N,N), periods=(N,N))`、
単位spacing、origin 0、Triweightの座標productである。
埋め込みTorus/Sphereの以前の時間とは異なる数学契約なので直接比較しない。

## 固定した条件

N=1024/2048、batch B=32、atom数 K=floor(0.05 N²)、seed=41、
初期rho=sigma/spacing=3/8。FP32 IEEE、TF32無効。
全候補で同じ初期atom・X・target・検証cotangent、宣言とbufferをhash照合する。
振幅・中心・共有live width、全productに一度だけ適用するL2 floorを共通にする。
task widthの微分はstop-gradientである。

同じAdamW（lr=1e-4、weight_decay=0.01、fused/capturable）とPolar更新、
周期中心retractionを使う。公開CSTOptimizerとの同一cotangent更新を別に検証する。
captureはGPU更新を実行しないため、2 eager warmup + 1 replay + 21測定replay
の実更新24回をoptimizerのcounterで確認する。

rho3は順序を反転した独立process比較をもう一度実行する。rho8は独立run 1回。
時間sampleを削除せず、各runの中央値と全sampleを保存する。

## 比較する実装

`research_cuda_periodic_grouped_matrix` は支持範囲を座標から算出し、
grouped atomic加算でWを構築する。Torch IEEE GEMMとTriton IEEE GEMMを別に測る。
backwardはdX=dY W、dW=dYᵀX、支持局所のatom VJPである。

`research_cuda_periodic_prepared_factor` は同じ準備情報を使い、
H=X Vᵀ、G=dY Uᵀを支持範囲だけで縮約する。出力/dXをgrouped atomic scatterし、
H/Gと局所微分からatom VJPを計算する。全siteのU/V行列は保存しない。
これは今回のH/G実装の比較であり、因数分解全般の限界を測るものではない。

公開Torch factored参照と通常のdense Linearも工学的な対照として測る。
denseは学習するParameterが異なる。Torch参照のOOMは条件を変えず記録する。

双方でW=Σₖ aₖ uₖ vₖᵀ、Y=XWᵀである。
生のprofileをqᵤ,qᵥとすると分母は
max(‖qᵤ‖₂ ‖qᵥ‖₂, floor)であり、軸ごとにfloorを掛けない。
全axisを列挙する独立FP64 oracleをatom chunkに分け、初期状態と更新24回後の
Y・dX・全4成分dPを照合する。max_absとrelative_l2の上限は各4e-4。

## 時間・メモリの意味

主指標は未計装のforward + loss + backward + optimizer全stepの同期wall時間。
capture/replayを含むallocated/reserved peakを、oracle scratchを解放してから測る。
各方式は独立processで動かし、CUDA allocator poolを共有しない。
これはallocator観測でありGPU process全体の使用量や物理DRAM/L2 trafficではない。

phaseイベント計測は主測定後の別の状態copyを使う。
forward/loss、backward、optimizerを計測し、別の固定状態forwardで
snapshot/prep、W組立またはH縮約、GEMMまたはscatterを診断する。
計装した各時間の和を主指標の代用にはしない。

FP32でWとdWは合計8N² byte、HとGは合計8KB byte。
K≈0.05N²、B=32では後者が前者の約1.6倍になる。
共通のpacked情報13K要素、Parameter/gradient/optimizer state、Graph poolも
ピークに含まれるので、実測の差をこの式だけで説明し切らない。

## 再現と適用範囲

```bash
python -m benchmarks.cuda.linear.periodic_comparison --state-gate --output output/periodic-state.json
python -m benchmarks.cuda.linear.periodic_comparison --size 1024 --rho 3 --phases --output output/periodic-1024-rho3.json
python -m benchmarks.cuda.linear.periodic_comparison --size 1024 --rho 3 --plans matrix-torch matrix-triton factor dense --reverse --output output/periodic-1024-rho3-repeat.json
```

この研究専用JSONはbenchmark提出schema/DB adapterの対象ではない。
新Algorithmはbenchmark-local Registryに登録し、公開dispatcherの既定選択は変えない。
今回の結論はcanonical中心と上記unit-spacing条件に限定する。
極端に大きな中心/原点のFP32距離計算は精度を失い得て、全axis fallbackは
探索漏れを防いでもその丸め誤差を修復しない。
