# owner-windowの同一状態・同期方式・checkpoint交差診断

PR #126の単層13.4〜13.6%短縮と、MNIST全体の改善なしを説明する短い診断。
新しいkernel最適化や既定dispatcherの変更は行わない。
元の結果は [followup記録](owner-window-followup-20261103.ja.md) を参照する。

**結果:** 同じcheckpointでも、毎step同期では併用版が6.61〜11.22%短縮したが、
256 step連続発行では一貫した短縮がなかった。学習後もGPU上の処理削減は残っており、
CPU発行を含むhost側とGPUの重なりが全体throughputを制約する説明を強く支持する。
数値差の入口はDFTのFP64 windowと微小なFP32 dPで、Adam momentsはstep1から分岐。
Parameterは確認した256 stepで一致し、45,000 step後のParameter差の発生点は未特定。

## 条件と順序

`benchmarks.training.owner_window_diagnostics` は既存MNISTのmodel、optimizer、data loader、
正規化を再利用する。初期値と両学習後checkpointをそのまま使い、長期学習をやり直さない。

1. 同じP/x/dYでbaseline/C/raw/DFT/combinedを比較。同一Plan再実行を対照とし、
   Y/dX/dP、nativeの入力window、全モデルのlogits/loss/grad、更新後の全状態を検査する。
   同じgradientを両Planへ与えた更新も比較。共有状態から短区間を逐次比較し、最初の更新差を
   発見したら、その直前のcheckpointとminibatchを保存して一回の再実行で照合する。
2. 同じモデル・全optimizer状態から、同じ256 minibatchを両Planで実行。
   毎step同期／区間同期の2方式、4反復、順序を反転して比較。主指標は区間wall timeの平均。
   初期performance状態だけはbaselineの共通8更新後。学習後の2状態は既存checkpointを使う。
   事前準備をwarmupし、Parameter/moments/stepを元のstorageへcopyして正確に復元する。
3. 両学習後checkpointをそれぞれ両Planで動かし、状態差と実装差を分ける。
   別passのCPU/CUDA profileでnative window kernelを確認し、timelineを保存する。

性能計測はeager/TF32 off、CPU math threads1、FP32 Parameter、内部FP64、真B4。
全train/test Tensorと全epoch用orderをGPUへ常駐させ、gather、cross entropy、全層backward、
実optimizer、GPU loss/accuracy集計を含む。評価・checkpoint復元・計装はwall timeから除く。
profileの時間を未計装の性能値と混ぜない。CUDA Event間隔にもCPU供給による空き時間が
入り得るため、それを純粋なGPU演算時間とは解釈しない。

native保存windowの未使用bankは未初期化なので比較せず、実際のnative keysが要求する
有効bankのFP64領域だけを照合する。CPU診断とnative owner keysの差も別に記録する。
profileではTorchDispatchMode等によるfallbackを避け、実際のkernel名を確認する。

## 準備時点の検証

CPU上で初期値・両checkpointをbaseline/combinedの各modelへ復元し、model/optimizerの
全Tensorが一致することを確認した。Parameter・exp_avgを意図的に変化させ、同一storageへ
復元する検査も通過。lint/formatを確認済み。これはGPU結果ではない。

入力checkpointとdatasetはignored outputへ保全し、pool driverへhash付きpayloadとして
含める。通常のsource snapshotへ非ignoredの生データを混ぜず、pool以外の直接uploadはしない。
driverとraw resultsは `output/owner-window-diagnostics/` とdefault pool job原本へ保持する。
結果・job/source/result hash・判断は実行後に追記する。

## 初回診断と検査側の修正

`l4job-5c7fc7f0698647c09bdbaa65085632e0` は正常終了し、数値比較は全条件を通過。
初期のDFT/combinedでは入力windowのmax_abs差4.663e-15、dP差2.776e-17が入り、
step1でAdam momentsが分岐した。初回Parameterは同じだった。同一gradientを使う更新、
同一Planの再実行、C/raw単独は全3状態でbyte一致。matched-stateの独立参照検査もPASS。
最初のstate差は直前の共通状態からbyte単位で再現できた。

初回の照合とメモリ値を点検し、次の検査側の問題を修正して再実行する。

- 既存 `radius_statistics` が入力/出力中心列を逆に読んでいた。正本READMEの列順は
  output→inputであり、1次元Polarでは出力列2、入力列3。CPU側owner keysの再構築にも
  同じ逆順を使ってしまっていた。演算本体や半径/bank数の診断は変更不要。
  4つの出力ownerと1つの入力ownerを持つ独立fixtureのCPU回帰検査を追加した。
- 多数のmodel/optimizerを作り直す診断の反復間で、循環参照の回収が一定でなかった。
  初回のallocated peakが約257〜438MBで変動したため、これを採用用memory値にしない。
  反復前後にgc.collectを行い、未計装の計測区間だけGCを固定して再測定する。
- 最初のAdam-state差と最初のParameter差を別々に追う。baseline自由走行の複製も置き、
  各stepで対照の一致を確認する。

初回profile全6条件でTorch dispatch stack深さ0、native owner-window kernelを確認した。
初回の数値結果とtraceは削除せず `output/owner-window-diagnostics/initial-results/` へ保全。
改善率とmemoryの確定表には修正後の比較を使う。

## 修正後のL4診断

`l4job-b065b3dcc24d4bc3a209704c42f65b74`、source
`eb81045268c9aea2a5be4bc9576437d5faf477f7` は正常終了した。
NVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、
Python3.13.16。CPU math threads1。GPU/hostは初回診断とは別VM。
CPU hostのbrand/quotaは収録していないため、他のhostへthroughputを一般化しない。
kernel/runtime sourceは元の測定 `bf52fb9…` から変更していない。

### 同一状態の数値比較

初期・現行学習後・併用学習後の3状態、5 Plan、各Planの再実行、
同じ勾配による更新の全対照が通過。C/raw単独は、局所Y/dX/dP/有効window/keys、
全モデルlogits/loss/grad、更新後のParameter/moments/stepが基準とbyte一致した。
全3状態でcombinedの同一状態CPU FP64参照もPASSし、相対L2誤差は約2.6e-8以下。

DFT単独とcombinedの差は同じだった。初期の局所比較は次のとおり。

| 比較対象 | 差のある要素数 | max_abs |
| --- | ---: | ---: |
| 有効なFP64入力window | 9,090 / 12,288 | 4.6629367e-15 |
| FP32全dP | 22 / 209,712 | 2.7755576e-17 |
| 最初のexp_avg | 20 / 209,712 | 3.4694470e-18 |
| 最初のexp_avg_sq | 22 / 209,712 | 2.5243549e-29 |

初回のY、dX、logits、loss、ordinary層のgrad、Parameter自体は一致。
同じgradientを与えたoptimizer更新は全状態・全Planでbyte一致し、更新則の違いを示す
証拠はなかった。各Plan自身の再実行もbyte一致した。

自由走行では毎stepを調べ、baselineの独立した複製も256 stepすべて一致した。
最初のAdam-state差はstep1で、直前の共通状態からの再実行で両側ともbyte一致して再現した。
一方、Parameterは256 stepを通して両Planで一致し、最初のParameter分岐は見つかっていない。
したがって「DFTの評価経路が微小なstate差の入口」という証拠は得たが、
45,000 step後の最大差0.05945までの増幅経路をすべて解明したとは主張しない。
その長期差を許容するかの判断も引き続き保留する。

### native keyと分布

修正後のCPU owner keyと実際のGPU keyは、3状態の全52,428 atomでbyte一致した。
GPUで直接数えたL12/L6も初期52,428/0、現行終点52,423/5、併用終点52,421/7。
保存状態でwide bankへの移行はなく、CPU分布からの推測だけでなくnative分類でも確認できた。
前記の中心列不具合は診断だけのもので、演算本体と半径/bank数には影響しない。

### 同期方式とcheckpointの交差比較

各セルは同じ256 minibatch、4反復の区間wall time平均。各checkpoint行の両Planは
全Parameter・buffer・optimizer stateを同じ状態へ復元してから比較した。
計測外のwarmup後に値を同じstorageへ戻し、開始状態のbyte一致も確認。
初期行だけはoptimizer cold startを除くため共通baselineの8更新後を使用した。

| checkpoint | 毎step同期 現行→併用 ms/step | 短縮 | 区間同期 現行→併用 ms/step | 短縮 |
| --- | ---: | ---: | ---: | ---: |
| 共通初期＋8更新 | 2.760647 → 2.578278 | 6.61% | 2.561875 → 2.550564 | 0.44% |
| 現行の学習後 | 2.915291 → 2.588171 | 11.22% | 2.507226 → 2.535637 | -1.13% |
| 併用の学習後 | 2.920131 → 2.627660 | 10.02% | 2.600054 → 2.599577 | 0.02% |

毎step同期の4 paired反復は全て併用版が速かった。区間同期は各checkpointで正負が混在し、
上表の小さい差を有意な改善・回帰とは扱わない。異なる終点同士を比べたことだけが
全モデルで利益が消えた理由ではなく、同じ状態でも同期方式で結果が変わると確認できた。

全48回のCST比較で開始allocated252,374,528 byte、peak allocated257,324,032 byte
（245.403320 MiB）、reserved436,207,616 byte（416 MiB）が同じだった。
これは短区間の計測範囲であり、初期評価等を含む元の3 epoch peak246.089844 MiBと
差し引いて追加のメモリ削減とは主張しない。process-wide memoryは未測定。

denseの同じ初期effective W／係数契約から8回自身で更新した参考値は、毎step同期
1.450833 ms、区間同期1.389248 ms（各1区間）。atomの学習後Adam stateと対応させた比較ではない。

### 別passのtimeline

6 profileすべてでTorch dispatch stack深さ0、native owner-window contractを確認。
combinedではbounded raw kernelも確認し、fallbackした経路の測定ではなかった。

| checkpoint | kernel実行union 現行→併用 ms/step | kernel数/step |
| --- | ---: | ---: |
| 共通初期＋8更新 | 2.104142 → 1.783492 | 両方90 |
| 現行の学習後 | 2.330352 → 1.912751 | 両方90 |
| 併用の学習後 | 2.361271 → 1.941127 | 両方90 |

同じ16 stepでCUDA API call数も両方2,338回、memset20回/step、memcpy3回/step。
device synchronizeはprofile区間全体で2回であり、毎batchの明示readbackを示すものではない。
memcpy/memsetも含むGPU activity unionは各Planとも上表より約0.0295 ms/step長い。
これらはlaunch数を減らさずGPUの仕事を短くした実装であることと整合する。

ただしprofileのspanは4.04〜5.14 ms/stepで、未計装の約2.5 msより明確に長い。
profileはCPU処理を増やすため、空き時間の量を未計装へ転用しない。
`profiles.json` のkernel間空白にはmemcpy/memsetも含まれるため、純粋なGPU idleとも呼ばない。
未計装の同期交差を主証拠、timelineを補助証拠として扱う。

## 判断・未解決点・証拠

学習後もGPU処理削減は残っていた。wide bankや極端なowner集中を原因とする証拠はなく、
同じ状態で毎step同期から連続発行へ変えると利益が消える。CPU発行・host側処理とGPUの
pipeline関係が主要な説明として残る。ただしCPUのどの関数が律速かは分離しておらず、
CPU律速の割合を定量化したとは言わない。

次の速度研究を行うならカーネル内の算術だけをさらに詰める前に、host発行やlaunch数を
減らす設計を検討する。今回それを追加実装したわけではない。Cだけの全モデル性能も未測定で、
combinedから推定して既定化しない。PR #126のDraft・採用保留は維持する。

- CPU標準suite: `1711 passed, 2494 skipped`。中心列の回帰検査を含むMNIST suite41件PASS。
- 修正後source archive SHA256: `491bf91bfc89e34fee34f7d8bcbd36ea599e84ff141b72103352947e882943d5`
- 修正後result archive SHA256: `23bc5c9fec9e5ab854404f1a39d4f495813fb85c8d32fba54fd6c19dffe27b3f`
- 初回source archive SHA256: `11c1b87a3dfc15c859ca42adb2eafdadca329c8d68e306e81aa9fada62e7bb71`
- 初回result archive SHA256: `b8839c3cf3b4724c726c071e3314fd0d52f2ec1dbae3ace25d0d6b48c3f359ae`
- 原本: default poolの両job directory。ローカル保全は
  `output/owner-window-diagnostics/{initial-results,controlled-results}/` と各spec/receipt。
- 要約: `timing-summary.json`、`profile-activity-summary.json`。元の48区間の値と全6traceも保持。
- 初期値・両checkpoint・datasetは同directoryの `inputs/`、hashは `input-manifest.json`。
  pool driverにも同じ入力をhash付きで含めてあり、source archiveだけでも再構成できる。
- 最初のstate差の再現情報は結果内 `numerical/first-state-difference-replay.pt`。
  初回差の直前は初期状態なのでcheckpoint欄はNoneで、同じinitialization.ptと組み合わせる。
- 両診断jobは回収・hash照合に成功し、supervisorは正常終了。所有slotは全てstoppedを確認。
  確認記録は `output/owner-window-diagnostics/final-pool-status.json` に保持。

再現用の短区間コマンド:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m benchmarks.training.owner_window_diagnostics \
  --inputs output/owner-window-diagnostics/inputs --stage numerical \
  --steps 256 --repeats 4 --output output/owner-window-diagnostics/numerical-replay
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m benchmarks.training.owner_window_diagnostics \
  --inputs output/owner-window-diagnostics/inputs --stage timing \
  --steps 256 --repeats 4 --output output/owner-window-diagnostics/timing-replay
```
