# raw階層のcooperative launch単独比較

PR #126後の診断で、GPU演算削減が連続発行する全モデルの短縮にならず、
両Planとも90 kernel/stepのままだった。今回は改善候補を1つだけ比較する。
作業branchは `kernel/owner-window-cooperative-raw`、親は `b9d53e71`。

## 候補と契約

基準 `GLOBAL_OWNER_WINDOW_RECOMPUTE` に対して、raw momentの9回/方向の起動を
1回/方向のcooperative kernelにまとめる。想定総起動数は90→74/step。
固定DFT、batch×feature C配置、wide-only mask変更は併用しない。

元と同じrequired-bank mask、raw配列shape、`compact_moment_element`、level順を使う。
次の有効levelがある時だけgrid全体を同期し、子momentの書き込み完了を保証する。
加算順、profile、半径・中心、正規化、optimizerの則は変えない。
raw配列も保存windowも削除しない。測るメモリはPyTorch allocatorの実ピーク。

cooperative launch対応をnative入口で確認し、実kernelのoccupancy×SM数からgrid上限を
計算する。全threadが同じmask/level条件でbarrierへ到達し、非一様な早期returnを作らない。
対応範囲は従来window Planと同じN1024、Polar/M8、同一軸、真B<=4。
未対応hardwareでは明示的に拒否し、既定dispatcherは変更しない。

根拠はNVIDIAの[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
と[cooperative launchのoccupancy上限](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/generated/driver_v2/cuda.bindings._v2.driver.launch_cooperative_kernel.html)。
速度改善はこの仕様から保証されるものではなく、同期・queryの追加費用も完全stepで評価する。

## 検証と比較手順

1. CPU宣言、checkpoint復元、lint、標準suite。
2. GPU独立FP64 oracle、B0/1/3/4、FP32/64、input-only/atom-only、広い半径・境界・
   owner集中、hook/retained/offload。3 stepのY/dX/dP/P/moments/stepのbyte gate。
3. 初期値と両学習後checkpointを両Planで比較。固定P/x/dYと全モデル・optimizer状態を
   一致させた検査を先行し、256 stepも状態のbyte一致を要求する。
4. 同じ256 minibatch、複数の順序反転反復、毎step同期と区間同期で全モデルを比較。
   本来の目的は区間同期のthroughput。別passでCUDA activityだけの軽いprofileも取り、
   native経路、実起動数、API呼び出し、計装の負荷を確認する。
5. 既存Linear runnerで基準/候補/denseを照合し、既存MNIST runnerで同一初期値・data orderの
   各3 epoch/45,000 stepを測る。最終model/optimizer状態、accuracy/loss、memoryを比較。

`benchmarks.training.owner_window_diagnostics --candidate GLOBAL_OWNER_WINDOW_COOPERATIVE_RAW`
で基準と本候補だけを選び、DFTを含む候補とは混ぜない。CPU brand/quotaも収録する。
raw inputsは親branchの診断と同じhash付き初期値・checkpoint・datasetで、pool driverへ含める。
生ログ・driver・結果はignored `output/owner-window-cooperative/` とdefault poolに保全する。
mainへの採用は測定後の判断で、ここではまだ行わない。

## 準備時点

宣言検査は31 Plan/本CaseでPASS。windowと本候補のCPU側検査は5 passed/273 skipped。
GPUでのcooperative launch成功、正しさ、速度、メモリは未確認。実行後にjob/hash/結果を追記する。

## 初回L4実測

`l4job-49bb39c1ef7546088396e730fd3025e3` は正常終了し、結果をhash照合して回収した。
測定sourceは `73488edf98e97b378256c768f594f6d0e5967a53`。
NVIDIA L4、driver580.82.07、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、Python3.13.16。
CPUはIntel Xeon @2.20GHz、12論理core、cgroup `max 100000`、math threads1。

GPU環境のunit検査278件が全て通過した。独立FP64参照、広い半径・周期境界・owner集中、
retained/offload/hook、3 stepのexact optimizer状態、全52,428 atomの更新を含む。
旧window3 Planも同じsourceで検査した。初期・両学習後checkpointの同一状態比較では、
Y/dX/dP/有効window/keys、全層grad、更新後の全model/optimizer Tensorがbyte一致。
256 stepの自由走行もParameter・Adam stateとも一致し、同一Plan対照も一致した。

### 連続発行する全モデルの短区間

同じ256 minibatch、各6反復で順番を釣り合わせた平均wall time。

| 固定checkpoint | 基準 ms/step | 候補 ms/step | 短縮 | paired反復の勝ち数 |
| --- | ---: | ---: | ---: | ---: |
| 初期＋共通8更新 | 2.488580 | 2.442404 | 1.86% | 5/6 |
| 現行の学習後 | 2.457750 | 2.394728 | 2.56% | 6/6 |
| 旧併用版の学習後 | 2.485434 | 2.456698 | 1.16% | 4/6 |

全比較でpeak allocated257,324,032 byte、reserved436,207,616 byteは同じ。
同じ初期状態からの別process比較であり、個々のstepを独立標本とは扱わない。

### 既存runnerによる完全stepと全MNIST

単層の毎step同期101標本の中央値は2.435501→2.402412 ms（約1.36%短縮）。
allocated7,456,768 byte／reserved8,388,608 byteは両方で同じ。
denseは0.712906 ms。単層だけの改善から全モデルの効果を推定しない。

既存MNIST runnerをそのまま使い、同一の初期値・effective W・データ順序、真B4、
各3 epoch/45,000 stepで基準→候補→denseの順に実行した。モデルは3つの重み付き層で、
CSTは中央の1024²層だけ。未計装の学習loop全体をepochごとに同期し、cold optimizerを含む。

| 全モデル | 平均 ms/step | 学習合計秒 | 最終test精度 | 最終test loss | allocated MiB | reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recompute | 2.487813 | 111.951564 | 97.37% | 0.082856 | 246.089844 | 392 |
| cooperative raw | 2.441883 | 109.884751 | 97.37% | 0.082856 | 246.089844 | 392 |
| dense | 1.266473 | 56.991270 | 97.87% | 0.070018 | 257.861328 | 384 |

候補は全モデルの3 epoch平均で1.846%短縮、epoch2–3だけでは2.417%短縮だった。
最終42 Tensorのmodel/optimizer状態が全てbyte一致し、全epochの学習・test指標も一致。
同じcheckpointを読み戻した全atomのY/dX/dP参照と層化8 atomの独立quadratureも両PlanでPASS。
データhash、正規化統計、初期値hash、各epochの順序hashも全runで一致した。

### profileの確認と限界

全6条件でnative window経路を確認し、dispatch stackは0。
実kernel数は90→74/step、16 stepのCUDA API記録数は2,338→2,146だった。
cooperative属性・occupancy queryが増えるため、API回数の減少はkernel数の差とは同じでない。

CUDA activityだけのprofileでもwall timeは約3.14〜3.57 ms/stepで、未計装値より長かった。
CPU eventを外しても計装費用は残り、その時間を性能値へ混ぜない。kernel実行unionは初期が
2.086909→2.095891 ms、現行終点が2.332209→2.318275 ms、旧併用終点が
2.337007→2.308529 msで、GPU演算そのものの大幅な短縮はなかった。
launchを減らす変更がhost発行側へ効く説明と整合するが、CPU関数ごとの未計装費用は未分離。

効果は小さいため、同じ候補を変更せず、別jobで候補→基準の逆順3 epoch比較を行う。
独立repeatのsourceも `73488edf…` で、ここまでの文書追記はそのsnapshotに含まれない。

- source archive SHA256: `2b75bdf028e5f1f67b04582570715de43f7f29d86f5e0e29b051af91af9a0f51`
- result archive SHA256: `de9974b4cdb16d15a7747eb166f7aad441552337c7246ec91cfdc8b447054619`
- 保全先: `output/owner-window-cooperative/first-results/` と `first-spec.json`、`first-receipt.json`。
- 原本: default pool job directory。raw inputsはdriverのhash付きpayloadにも含む。
- CPU標準suiteは `1712 passed, 2599 skipped`。宣言・lint・checkpoint復元もPASS。
