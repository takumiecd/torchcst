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
