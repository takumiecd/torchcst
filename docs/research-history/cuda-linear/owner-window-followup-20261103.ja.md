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

### 演算を変えない範囲

L12の実基底を $B_0(j)=1$、$B_{2k-1}(j)=\cos(2\pi kj/12)$、
$B_{2k}(j)=\sin(2\pi kj/12)$ とする。$k=6$ はcosineだけを保持する。
owner窓のmomentと出力展開は、それぞれ

\[
Z_{bof}=\sum_{j=0}^{11}X_{b,(4o+j)\bmod N}B_f(j),\qquad
Y_{b,(4o+j)\bmod N}\mathrel{+}=\sum_f C_{bof}B_f(j).
\]

固定tableはこの $B$ の値をFP64で一度丸めて保持する。逆DFTによる復元ではなく
既存の基底和なので、追加の $1/12$ は掛けない。任意中心・半径から得る係数 $q$
は従来どおり毎回計算し、cell integrationのsincも一度だけ掛ける。
中心微分にはalias後の番号でなく元の $\omega_k=2\pi k/P$ を使う。

C縮約は $C_{bof}=\sum_{a\in\mathcal A_o}q_{af}(A_aH_{ba})$。
各 $(b,f)$ のatom/chunk順を同じに保って担当threadだけを変える。
raw側はwide consumer集合から必要な下位bankを展開した和集合を計算する。
短窓bankを一律に取り除く方法では、wide consumerの子momentを欠落させる。

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

## L4 初回比較

source commit `bf52fb933dd387d815395d312fd9c0c5dfbd7cfa`。
`l4job-9a4503d3113b48139b0272bb654e4c0c` は成功し、結果をhash照合して回収した。
NVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、CPU 12論理core、
Torch/OMP/MKL/OpenBLASのthread数1。TF32 off、FP32入出力・Parameter、内部FP64。
N1024/A52428/真B4、native CSTAdamW、dX込み、Graphなし。

C専用testは106件、followup+旧window testは449件、計555件が全て通過。
全6 Planが既存Linear runnerの全atom Y/dX/dP、独立quadrature、3 stepの
matched optimizer状態検証と、初期・更新後の全atom検証を通過した。
全atomの初期max_abs誤差はY `1.145e-10`、dX `1.162e-10`、dP `1.189e-7` 以下。
配置変更のbyte gateと固定DFTを含む一般の数値gateは区別する。

未計装の完全step、warmup10回、計測101回の中央値。別に同期したphase計測は
原因の参考に留め、完全stepから差し引いたり足し合わせたりしない。

| Plan | 完全step ms | 基準からの短縮 | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| recompute | 2.434310 | — | 7.111328 | 8 |
| c-batch | 2.318125 | 4.77% | 7.111328 | 8 |
| fixed-dft | 2.254450 | 7.39% | 7.111328 | 8 |
| tight-raw | 2.405737 | 1.17% | 7.111328 | 8 |
| combined | 2.103889 | 13.57% | 7.111328 | 8 |
| dft-raw | 2.220495 | 8.78% | 7.111328 | 8 |
| dense | 0.704136 | 別model/更新 | 32.329590 | 42 |

先に実行したCだけの比較でもrecompute `2.451361 ms`、c-batch `2.328365 ms`
（5.02%短縮）で、allocated/reservedは同じだった。同一VM内の別process比較であり、
独立VMの再現2回とは数えない。combinedはdenseの約2.99倍で、denseに並んだとは言えない。

`cuobjdump`で固定DFT展開のregisterは42→37、C forwardは106→108、backwardは125で同じ。
Cのstack frameはforward1056→1024 byte、backward1088→1056 byte。
これだけでspillやoccupancyの改善を断定しない。dynamic sharedは22,048 byteで同じ。

このfixtureは研究専用で、既存submission/database adapterが明示的に受付対象外とする。
ローカルsubmission検査のrejectはこの既知の境界であり、GPU演算検証の失敗ではない。
今回DBやIssueへの投稿は行っていない。

- source archive SHA256: `3a9396209ec51561d323434f16a9c7291873fad592d789f85b7f18cb5cb57648`
- result archive SHA256: `a537ab99a5a458b5ae5b482daa02e987c7ecbeb3a75d7d3bdb81b22303ee95eb`
- native binary SHA256: `42f0329ef1bcbb5012372e9b081ad9cb187e720ad4bd238f683e24bf6ceefb10`
- local保全: `output/owner-window-followup/initial-results/`、`initial-spec.json`、`initial-receipt.json`
- pool原本: `/Users/ware10sai/.local/state/colab-l4-pool/jobs/l4job-9a4503d3113b48139b0272bb654e4c0c/`

初回最速のcombinedを選び、順番を反転したfresh processの再測定と、同一初期値・データ順序の
baseline/candidate/denseのMNIST各3 epochを次のjobで実行する。1 seedであり、
学習一般化の証明ではない。resident dataを含むMLP全体のmemoryを改めて測り、
単層の44%削減率をそのままモデル全体へ転用しない。
