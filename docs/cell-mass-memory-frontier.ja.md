# Cell-mass: 完全stepのメモリ制約と速度

ユーザーの制約は「dense以下のメモリ」、時間目標は「denseの約2倍以内のstep時間」。
128×128 / batch 4では両立したが、batch 16へそのまま一般化できない。
保存を増やせば必ず速くなるわけでもない。

## 比較条件

L4、PyTorch 2.11、FP32、TF32無効、Graphなし。819 atoms（約5%）、M=8、
seed=120、入力tile16 / 出力tile32、sigma初期値0.2..12。
固定X/dYに対するforward・backward・fused AdamW更新を含む。
各設定を独立した2プロセスで測定、2回目は順番を逆転、warmup10 / 同期wall時間30標本。
[projection cacheの数式と検証](cell-mass-projection-cache.ja.md)と同じruntime。
公開dispatcherへの採用は未実施。

cuBLASのworkspaceによる見かけの差を避けるため、2種類のdenseを測った。

- 時間目標の基準: 通常の高速なdense（workspace設定なし）。
- 保守的なメモリ上限: `CUBLAS_WORKSPACE_CONFIG=:0:0` と
  `CUBLASLT_WORKSPACE_SIZE=0` のdense。

この2つは別の実行条件であり、一つのdense実行の時間・メモリとは区別する。
CSTはworkspaceなしの条件で測定した。モデルの学習パラメータはdenseではW、
CSTではphysical atomsであり、更新後の学習軌跡の一致は主張しない。

## 実測

時間は独立runそれぞれの中央値。メモリは完全stepのallocatorピーク。

| batch | 設定 | step ms (run 0 / 1) | allocated KiB | reserved MiB | workspaceなしdense以下 |
|---:|---|---:|---:|---:|---|
| 4 | dense通常 | 0.450 / 0.439 | 16904.5 | 22 | 基準とは別条件 |
| 4 | dense workspaceなし | 0.641 / 0.633 | 264.5 | 2 | 上限 |
| 4 | 再計算 split4 | 1.074 / 1.082 | 110 | 2 | ○ |
| 4 | 再計算 split8 | 0.969 / 0.972 | 113 | 2 | ○ |
| 4 | 微分保存 split4 | 0.940 / 0.940 | 197 | 2 | ○ |
| 4 | 微分保存 split8 | 0.835 / 0.837 | 200 | 2 | ○ |
| 16 | dense通常 | 0.444 / 0.461 | 16928.5 | 22 | 基準とは別条件 |
| 16 | dense workspaceなし | 0.640 / 0.662 | 288.5 | 2 | 上限 |
| 16 | 再計算 split4 | 1.839 / 1.838 | 279 | 2 | ○ |
| 16 | 再計算 split8 | 1.844 / 1.848 | 311 | 2 | × |
| 16 | 微分保存 split4 | 1.178 / 1.181 | 623 | 2 | × |
| 16 | 微分保存 split8 | 1.190 / 1.192 | 655 | 2 | × |

batch 4の微分保存split8は通常denseの1.86 / 1.91倍の時間で、
allocated200 KiBはworkspaceなしdense264.5 KiBの約76%。
batch 16では保存4本が4×batch×atoms×8 bytes増えるため、保存案は上限を超える。
再計算split4だけが今回試したCSTの中で制約を満たし、時間は通常denseの約4.1倍。
こちらは速度目標未達。splitを4から8へ増やしても改善しなかった。

既定値の`cache_projection_derivatives=False`は維持する。
「キャッシュを常に有効化する」は採用せず、batch・atom数・split数を含む実測で判断する。
速度とメモリは単調な関係ではなく、短いreduction、再計算、保存の交換条件となる。

## denseの大きな割当の確認

計測後の隔離プロセスで`torch._C._cuda_clearCublasWorkspaces()`を呼ぶと、
allocatedが17,240,576から201,216 bytesへ減り、差は17,039,360 bytes（16.25 MiB）。
その後`empty_cache()`でreservedも22から2 MiBへ減った。
この操作は時間測定中には行っていない。
[PyTorch 2.11のworkspace仕様](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cublas-workspaces)
とも一致する。モデル本体の圧縮率としてこの16.25 MiBを数えない。
プロセス全体のGPUメモリ、CUDA context、driver側の割当はこのallocator値に含めていない。

## 証拠と再現

job `l4job-e894356533a34250a26aef80bf7d9e9b`。
凍結runtimeは`4c435bfd`と321ファイルbyte一致。
結果manifest157ファイル、埋め込みhelper、実行binaryも照合済み。
batch4/16、default/zero-workspaceそれぞれでY/dX/全atom勾配・
同一GPU fused AdamWへoracle勾配を渡した更新量とmomentsを検証した。
既存309件のGPU回帰はruntime不変のため繰り返していない。

- source SHA256: `0ebee49e51be97f3982bdacdcfa5378748873d31351dd54f73e09213cc0f42a9`
- result SHA256: `99d95f40f6cefb320dde40e30fc8cd0385dc43f061d9f0bcb50d94027f003816`
- driver SHA256: `7d1f5fa0814c434a8180e948532c14e9618f19d59dbf61423f7556521ac25a69`
- binary SHA256: `32ed229a350aef588151e8b5ffd42b66265e411cd774af13bc5555e7002e7076`

raw logs / source / archives / JSON / scatter plotはignored
`output/cell-mass-memory-frontier-20261027/`に保持する。
元source archiveを展開し、`src/`とrepo rootをPYTHONPATHに追加して、
`CST_JOB_OUTPUT`を空directoryに設定し`__pool_driver__.py`を実行する。
L4共有poolへの再提出は同directoryの`driver.py`を使用する。

この結果は128サイトの測定であり、1024サイトは別途検証する。
