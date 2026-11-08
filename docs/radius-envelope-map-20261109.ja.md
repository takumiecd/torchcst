# 半径写像の定数準備パイロット

[English](radius-envelope-map-20261109.md) | **日本語**

明示的な候補`RADIUS_ENVELOPE_CACHED_LM`は、宣言の定数と物理OperatorSpecを
bindingの配置ごとに一度準備する。非有限値、範囲外のradius_fraction・振幅は、
統合した1回の同期検査でnative実行前に拒否する。FP64の写像計算、FP32物理行、
端点の内向き微分、すべてのタスク勾配を維持する。学習する行・入力・出力・
各呼出しのautogradグラフを再利用状態に保存しない。旧`RADIUS_ENVELOPE_LM`を
比較対象として維持し、どちらも公開デフォルトには採用していない。

RTX 6000 Ada、Torch2.9.1/cu128で、同一ジョブ内の順序反転と各条件の新規workerを
使用した。R256/A8192、M8、tau.03、fraction0、半径[.5,50]、FP32、TF32無効。
既存のradius学習ループでzero_grad、入力勾配消去、MSE前後伝播、AdamW、任意で
公開CST射影・状態輸送を含む。比較はLR0、CSTは正のsub-ULP LRで物理状態を固定。
5warmup・同期した21標本であり、21回の独立実験とは数えない。

|方向|旧写像ms|定数準備ms|旧+CSTAdamW ms|定数準備+CSTAdamW ms|
|---|---:|---:|---:|---:|
|512→2048|3.961–3.991|3.840–3.840|3.979–3.998|3.842–3.854|
|2048→512|3.984–3.990|3.854–3.866|3.980–3.996|3.838–3.844|

このパイロットでは約3〜4%短縮。公開の速度採用を示すものではなく、小差の確証には
別ジョブでの再測定が必要。物理半径直接は3.278〜3.293ms、Denseは.217〜.317msで、
この小規模でのDense優位は残る。allocated peakは定数準備でup/down69.693/68.193MiB、
旧69.504/68.004。CSTAdamW付きで69.443/67.943、旧69.504/67.879。
reserved80/82MiBは同じ。定数準備は時間標本に含まず、warmupとmomentsはピークに含む。
Graph・プロセス全体のピークは未測定。長時間のリークについては主張しない。

実CUDAのY/dX/全フィールドVJP、retained backward、保存復元後の次更新の完全一致を
両経路で含む57検査が成功。CPU一式2241成功/2621skip。
[JSON記録](radius-envelope-map-20261109.json)にsource・job・archiveのhashを記載。
証拠53ファイルと凍結source1313ファイルを検証し、記録した26プロセスは回収後に停止済み。
再現は`benchmarks.training.radius_envelope_probe --arm cached`または`cached_cst`に
`--ni 512 --no 2048 --output ...`を渡す（downは次元を逆転）。
物理状態固定の測定であり、LMの収束や通常LRの学習結果ではない。
