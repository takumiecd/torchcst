# 等間隔PeriodicGridのgrouped Wと支持局所H/Gの比較

## 2026-10-10 L4結果

この範囲ではgrouped W + Torch GEMMが最速だった。
支持局所H/Gは約3.1〜3.9倍遅く、2048²ではallocated peakも増加した。
メモリを優先するならW + Triton GEMMが候補になる。
H/Gは比較用の研究候補として残し、公開dispatcherへの採用はしない。

各セルは **完全step中央値ms / allocated MiB / reserved MiB**。
rho3はrun1を示し、その下に順序反転run2も記録する。

| N / 初期rho | W + Torch GEMM | W + Triton GEMM | 支持局所H/G | Torch factored参照 | dense対照 |
| --- | --- | --- | --- | --- | --- |
| 1024 / 3 | 0.393 / 48.7 / 96 | 0.463 / 32.4 / 56 | 1.278 / 37.2 / 56 | 159.556 / 3110.5 / 3188 | 0.085 / 33.0 / 86 |
| 1024 / 8 | 0.869 / 48.7 / 96 | 0.942 / 32.4 / 56 | 2.727 / 37.2 / 56 | 159.738 / 3110.5 / 3188 | 0.085 / 33.0 / 86 |
| 2048 / 3 | 1.326 / 96.2 / 162 | 1.507 / 79.4 / 162 | 5.168 / 99.4 / 170 | OOM: initial-full-oracle | 0.594 / 81.8 / 106 |
| 2048 / 8 | 3.396 / 96.2 / 162 | 3.541 / 79.4 / 162 | 11.387 / 99.4 / 170 | OOM: initial-full-oracle | 0.594 / 81.8 / 106 |

順序反転run2のrho3中央値は、1024でTorch W 0.392377 / Triton W 0.462358 /
H/G 1.278417ms、2048で1.324883 / 1.507725 / 5.180840ms。
allocated/reservedはrun1と同じで、H/Gが遅い結論も変わらない。
各runは別processの21sample中央値であり、複数runから最速sampleを選んでいない。

別の固定状態forwardで計測した中央値から、次を優先する。

- **広いrhoのW組立**: 2048/rho8のWゼロ初期化＋組立1.891msに対し、
  forward Torch GEMMは0.033ms。次の速度改善は支持patchの組立と加算の分解を狙う。
- **狭いrhoのprep・更新・VJP**: 2048/rho3ではsnapshot/prep 0.268ms、
  Wゼロ初期化＋組立0.298ms、全optimizer phase 0.306ms、全backward phase 0.381ms。
  GEMMだけを置き換えても改善余地は小さい。prepの融合やPolar/周期更新の融合、
  backwardの内訳を次の候補ごとに確認する。
- **H/Gのscatter**: 2048/rho8のH縮約0.616msに対しY scatter 3.774ms。
  2048/rho3でもH 0.312ms / scatter 1.492ms。現在のgrouped scatterが重い。
  siteの所有や加算の集約を変える案は有望だが、原因をL2 missやatomic競合と
  物理的に特定するにはtrace/カウンタ計測が別途必要である。

W + TorchはTritonより約16〜17MiB多いallocatedが観測された。
cuBLAS workspaceが候補だが、allocator attributionは未実施であり確定しない。
単純なW/dW対H/Gの要素数だけでは、この差もGraph poolの差も説明し切れない。

検証はCUDA **72 passed**、CPU **1480 passed / 2702 skipped**、Ruffとwheel/sdist build PASS。
独立FP64 oracleは初期・24更新後の全Y/dX/全4成分dPでPASS。
最大観測max_absは約3.36e-4、relative_l2は約1.01e-6で、事前固定gateを変更していない。
全atomのwidthが変化し、public optimizerとの同一cotangent20更新は
Parameter/momentsに誤差0、counterは一歩検証を含め21で一致した。

測定sourceは `bd190113699f925fa97dd10974c3f8a776bcb333`、
jobは `l4job-066e7dcf48e14079a47916753dd66a7c`。
hardwareはNVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、
Triton3.6.0、Python3.13.15。
source archive SHA256は `2cbcf6649f57954259321b427460b039fa86a79c141b43de96e3c8b86b69a3d1`、
raw receipt SHA256は `315686f99b37100025559e22c316b2588b21cdacf72e12944d84df649f032cdc`。
全receipt hashと報告source file hashを手元で再照合した。
原本は共有poolの同job `results/artifacts/`、copyと集計は研究worktreeの
ignored `output/periodic-comparison/measured-bd190113/` に保存している。
開発時のclock配置テスト失敗2件を含む前job
`l4job-443b701edcbb4a2a9f8acdd64454e105` も保持する。

## 比較の契約

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
