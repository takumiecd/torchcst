# PR #125 後の3案の単独・組み合わせ検証（2026-11-03）

基準は `7aa7c89e42a4dd515630f8358e4ade880ddea482` の
`GLOBAL_OWNER_WINDOW_RECOMPUTE`。研究 branch は `kernel/owner-window-followup`。
worktree は `/Users/ware10sai/.codex/worktrees/owner-window-followup/torchcst`。
既存 Plan の既定、自動選択、main を変更せず検証する。

| mode | Plan suffix | C の batch×feature 配置 | 固定12点DFT | wide raw mask + level別grid |
| ---: | --- | --- | --- | --- |
| 2 | recompute | 既存 | 既存 | 既存 |
| 3 | c-batch | 有効 | 既存 | 既存 |
| 4 | fixed-dft | 既存 | 有効 | 既存 |
| 5 | tight-raw | 既存 | 既存 | 有効 |
| 6 | combined | 有効 | 有効 | 有効 |
| 7 | dft-raw | 既存 | 有効 | 有効 |

C 縮約は各 (batch,feature) を別threadが担当し、atom/chunkの加算順を維持する。
B4/r12では参加threadが12→48。r17でも68 threadで既存の128 thread CTA内に収まる。
保存Tensorやglobal Cの形を変えない。

固定DFTはL12だけを対象とする。12乗根の対称性（0、±1/2、±sqrt(3)/2、±1）から
144個のFP64定数を作り、window moment生成と展開で使う。ほかのbankは既存経路。
中心と半径に依存する係数は毎回生成し、元の周波数で中心微分してからfoldする。
profile係数の補間やM削減は行わない。cell integrationのsincを二重に適用しない。

raw maskはまずL>16のconsumer bankを選び、その後に下位依存bankを展開する。
短窓bankを依存関係ごと削除しない。raw level kernelはglobal Zと同じ位置へ書き、
同じ `compact_moment_element` で計算する。9 levelの合計gridはB4で4887→546 CTA。
raw配列と18回/完全stepのlevel launch自体は残る。配列やlaunchを消したとは主張しない。

## 検証の境界

- N1024、同一Polar/M8/base1/merge2軸、B<=4。公開の対応範囲は拡張しない。
- 独立CPU FP64 oracleでY/dX/全atom勾配、境界・重複像・owner偏り・retained/offloadを検査。
- Cとrawの配置変更は3 stepのY/dX/dP/Parameter/moments/stepのbyte一致も検査する。
- FP32更新はnative CSTAdamW。FP64のradius更新はGPU非対応なので、FP64 GPU Linearからの
  勾配をCPU CSTOptimizer参照へ渡し、更新PをGPUへ戻して次stepを検査する。
- 固定DFTの一般の契約は既存の数値基準で照合し、全条件のbyte一致を前提にしない。
- 全52,428 atomのrunner検証、独立quadrature、matched optimizer、同GPU wrapper byte gateを通す。
- 完全stepとpeak allocated/reservedを同条件で比較。Graph非対応、process memory未測定。
- 一つずつの変更を比較後、有効な組み合わせを選び、長期MNISTで品質とモデル全体を確認する。

## CPU 検証

標準suite: `1710 passed, 2494 skipped`。宣言、lintと実C++ helperも通過。
実際の新raw level kernelを独立threadとしてCPUで実行し、全bank/level、B4、spacing .37、
非零originで旧moment構築と全69,496要素がbyte一致。起動546 CTAも確認した。
これはCUDA実行時間やGPUのFMA命令の一致の証拠ではない。

## Colab の試行履歴

C単独 branch: `/Users/ware10sai/.codex/worktrees/owner-window-c-batch/torchcst`。

- `l4job-3e780ce504f54158a74df62ba3052613`: 独立VJP等58件は通過。native FP64更新を指定した
  検査設定が既存のFP32専用条件に反した。性能測定には進んでいない。
- `l4job-6136e0061f3b4af4a8cf29492a7e0e98`: Colab CLIの完了応答取得が1740秒でtimeout。
  結果archive未回収。poolが所有VMを停止し、serverからの消滅を確認。recover完了。
  演算性能の負の結果として数えない。
- `l4job-1a932f61437c477bb10d5690b5d33a52`: 診断logで58件通過後、FP64 Torch radius更新も
  CPU専用であることを確認。GPU上で参照更新を呼ぶ検査設定を修正した。
  CPU FP64 CSTOptimizerの3 stepはローカルで通過。性能値はない。

以後はCPU参照を対応device上で実行する。CPU thread数を1に固定し、CXX=g++、
個別コマンドの時間制限・phase時間・pytest診断logを記録する。
異なるCPU thread方針の過去の絶対時間から速度比を計算せず、同一job内の基準/候補で比較する。

## 再現と保全

```bash
PYTHONPATH=src python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-followup-initial.json
PYTHONPATH=src python -m pytest -q tests/test_radius_owner_window_c_batch.py tests/test_radius_owner_window_followup.py tests/test_radius_owner_window.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-followup-initial.json \
  --polar-update radius --phase-diagnostics --output output/owner-window-followup/benchmark.json
```

ignored driver・ログ・保全先は `output/owner-window-followup/`。pool job原本も保持する。
GPU結果、job/source/result hash、採否、長期学習結果は完了後に追記する。
