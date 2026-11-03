# owner C 縮約の batch×feature 配置（2026-11-03）

基準は PR #125 の merge `7aa7c89e42a4dd515630f8358e4ade880ddea482`。
研究 branch は `kernel/owner-window-c-batch`、worktree は
`/Users/ware10sai/.codex/worktrees/owner-window-c-batch/torchcst`。
既定 Plan を変更せず、`GLOBAL_OWNER_WINDOW_C_BATCH` を明示的候補にする。

## 検証する変更

旧 `GLOBAL_OWNER_WINDOW_RECOMPUTE` は thread が feature f を担当し、batch b を内側で巡回する。
新候補は `thread = b * rank + f` として、C の各 (b,f) を別の thread が計算する。
B4/r12 では C 縮約に参加する thread は12→48。
各出力の chunk/atom 順は同じとし、atom 方向の並列和や加算順の変更は導入しない。

$$
C_{b,o,f}=\sum_{a\in o}\alpha_a H_{b,a}q_{a,f}.
$$

変更箇所は forward/reverse owner CTA 内の C 縮約と書き出しの配置だけ。
profile、fold、中心微分、H/Hμ 再計算、Z/window producer、分類・sort、optimizer は共通。
`if constexpr` で基準と候補を独立にコンパイルする。
新候補の accumulator は thread ごとに1要素、基準は従来どおり4要素。
全体 Tensor の容量と保存 payload は共通。Torch allocator peak の実測を別途行う。

続編の CPU 検算では224条件の scalar 加算順一致を確認したが、GPU のFMAや速度の証拠ではない。
実GPUでは、FP32/FP64・B=0/1/3/4・joint/input-only/atom-only・通常/129 atom owner偏りの
48条件で3 stepの Y/dX/dP/Parameter/optimizer moments/step を byte 比較する。
FP32 は native CSTAdamW、FP64 は既存 CSTOptimizer/Torch 更新をCPUで使用する。
後者もLinear自体はGPUで実行し、GPU勾配をCPU参照へ渡し、更新ParameterをGPUへ戻す。
native radius更新はFP32専用、FP64 Torch radius更新はCPU専用である。
別途、独立 CPU oracle、retained/offloaded saved Tensor、hook/lifetime/versionの検査を行う。

## 比較条件と再現

N1024、B4、A52,428、同一入力/初期Parameter/target/native CSTAdamW。
基準・候補・dense を既存 runner で比較する。10 warmup、101 timing samples。
Graph は既存 radius と同様に非対応であり、capture/replay の測定値はない。

```bash
PYTHONPATH=src python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-c-batch-initial.json
PYTHONPATH=src python -m pytest -q tests/test_radius_owner_window_c_batch.py tests/test_radius_owner_window.py
PYTHONPATH=src python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-c-batch-initial.json \
  --polar-update radius --phase-diagnostics --output output/owner-window-c-batch/benchmark.json
```

ignored driver・ログ・保全先は `output/owner-window-c-batch/`。
原本は default Colab L4 pool の job directory に残す。
DFT定数化・raw moment削減・長期MNISTはこの候補へ混ぜない。

## 結果

followup branchへ合流したsource `bf52fb933dd387d815395d312fd9c0c5dfbd7cfa` で、
C専用GPU検証106件が全て通過。上記48条件のbyte比較、全52,428 atomの
FP32 native CSTAdamW 3 stepのbyte比較も通過した。

同一L4での独立process比較は基準 `2.451361 ms` → C配置 `2.328365 ms`（5.02%短縮）。
別の全候補比較processでも `2.434310 ms` → `2.318125 ms`（4.77%短縮）。
どちらもpeak allocated `7,456,768 byte`、reserved `8,388,608 byte` で同じ。
同一job/VM内での2比較であり、独立VMの2 runとは数えない。
job `l4job-9a4503d3113b48139b0272bb654e4c0c`。
hash・失敗試行・後続のDFT/raw/長期学習は
[followup記録](owner-window-followup-20261103.ja.md)へ集約する。
