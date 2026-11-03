# PR #125 後の3案の単独・組み合わせ検証（2026-11-03）

基準は `7aa7c89e42a4dd515630f8358e4ade880ddea482` の
`GLOBAL_OWNER_WINDOW_RECOMPUTE`。研究 branch は `kernel/owner-window-followup`。
worktree は `/Users/ware10sai/.codex/worktrees/owner-window-followup/torchcst`。
既存 Plan の既定、自動選択、main を変更せず検証する。

**結論:** 3案併用は単層の完全stepを2回のjobで13.4〜13.6%短縮し、ピークmemoryも維持した。
一方、matched MNIST 3 epochのモデル全体では基準2.496301→候補2.515619 ms/stepで、
短縮を観測しなかった。数値契約は通過したが最終学習状態はbyte一致しない。
追加のmain統合は保留し、実装・比較・checkpointを研究branchとignored evidenceへ保持する。

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
さらに今回の全候補JSONは49,852,422 byteで、submission CLIの5 MiB上限も超える。
ローカルsubmission検査はrejectされた。これはGPU演算検証とは別の提出条件であり、
adapterの対象範囲を今回拡張しない。DBやIssueへの投稿は行っていない。

- source archive SHA256: `3a9396209ec51561d323434f16a9c7291873fad592d789f85b7f18cb5cb57648`
- result archive SHA256: `a537ab99a5a458b5ae5b482daa02e987c7ecbeb3a75d7d3bdb81b22303ee95eb`
- native binary SHA256: `42f0329ef1bcbb5012372e9b081ad9cb187e720ad4bd238f683e24bf6ceefb10`
- local保全: `output/owner-window-followup/initial-results/`、`initial-spec.json`、`initial-receipt.json`
- pool原本: `/Users/ware10sai/.local/state/colab-l4-pool/jobs/l4job-9a4503d3113b48139b0272bb654e4c0c/`

初回最速のcombinedを選び、順番を反転したfresh processの再測定と、同一初期値・データ順序の
baseline/candidate/denseのMNIST各3 epochを次のjobで実行する。1 seedであり、
学習一般化の証明ではない。resident dataを含むMLP全体のmemoryを改めて測り、
単層の44%削減率をそのままモデル全体へ転用しない。

## 再測定とMNIST全モデルの結果

`l4job-8fb5b173d7104d1aa75e1eec9d540d4e` は正常終了し、全結果をhash照合して回収した。
同じL4 VMを再利用した別jobで、全workerと学習はそれぞれfresh process。
source commitは両jobとも `bf52fb933dd387d815395d312fd9c0c5dfbd7cfa`。
前節の結果を追記した文書commitは実行snapshotに含まれない。

Plan順を反転した再測定も全gateがPASSし、単層recompute `2.443231 ms` →
combined `2.115821 ms`（13.40%短縮）。両方allocated `7,456,768 byte`、reserved
`8,388,608 byte`。denseは `0.698996 ms`。単層の改善は再現した。

MNISTは既存 `benchmarks.training.cell_mass_mnist` を変更せず使用した。
`Linear(784,1024) → ReLU → CST/dense(1024,1024) → ReLU → Linear(1024,10)`、
重み付き層は3つでCSTは中央の1層だけ。全train60,000/test10,000、seed41、真B4、
各3 epoch/45,000 step。同一初期effective W・入力/出力層・データ順序・optimizer係数。
常駐データ、loss、全層のbackward、実optimizer更新を含み、Graphなし、TF32 off。
時間はepochごとの同期を挟む学習loopのwall timeであり、評価・checkpoint保存を除く。
epoch1にはoptimizer cold startを含める。単層の毎step同期プロトコルと区別する。

| 全モデル | 3 epoch平均 ms/step | epoch2–3平均 ms/step | 最終test accuracy | 最終test loss | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recompute | 2.496301 | 2.468442 | 97.37% | 0.082856 | 246.089844 | 392 |
| combined | 2.515619 | 2.484412 | 97.39% | 0.083560 | 246.089844 | 392 |
| dense | 1.343577 | 1.352646 | 97.87% | 0.070018 | 257.861328 | 384 |

候補の全学習時間は基準112.333532秒に対して113.202859秒、0.77%長かった。
1回のpaired学習結果であり、この小さい差から有意な遅延を断定しないが、全モデルでの
高速化を主張する証拠はない。allocated/reservedも追加で減っていない。
全モデルのCST/dense比は基準約1.86、候補約1.87で、候補だけが新しく達成した値ではない。
同一精度のモデルになったとは言わず、上記の1 seedのloss/accuracyをそのまま残す。

42 Tensorの最終model/optimizer状態は全てfiniteで、うち17 Tensorが基準とbyte不一致。
atom Parameterのmax_abs差は0.0594482421875、入力層weightは0.06515115150250494。
固定DFTを含む候補の45,000 stepをbitwise同一軌跡とは扱わない。
初期状態での丸め差の長期蓄積や各演算の非決定性をこの比較だけで切り分けていない。
これはC単独/raw単独の3 step byte gateが失敗したという意味ではない。

学習後checkpointを読み戻し、基準/候補それぞれの全52,428 atomについてCPU FP64
Fourier参照とY/dX/dPを検査してPASS。候補max_absはY `1.489e-8`、dX `2.840e-8`、
dP `1.188e-7` 以下。各checkpointから半径で層化した8 atomは独立48点quadratureもPASS。
入力・半径・中心が動いた後も数学・微分の検証を通過した。

単層と全モデルで差が消える原因は未診断。各step同期の有無、CPUの発行費用、overlap、
学習後の支持分布などを分離したprofileは取得していない。原因を決め打ちして
追加の最適化を重ねず、今回の結果では採用保留とする。mainはPR #125の状態を維持。

- initialization SHA256: `b49e95e722dab4e42d935a9ac2b239212e93660af55bf6d6a9f12f437a9f2ed6`
- source archive SHA256: `e319198d32178d9664011c5ebe90c3e2c4e3c55178c0e2661e1d9b2ea325ee66`
- result archive SHA256: `2fa0da8ca94b21e06d167831cfccbd6ccabb7f8f7f8281645e066377ab167e15`
- local保全: `output/owner-window-followup/mnist-results/`、`mnist-spec.json`、`mnist-receipt.json`
- pool原本: `/Users/ware10sai/.local/state/colab-l4-pool/jobs/l4job-8fb5b173d7104d1aa75e1eec9d540d4e/`

各epochのdata order hash、dataset MD5/SHA256、Plan snapshot、初期値、最終checkpoint、
全学習metrics・状態差・最終oracle検査は上記raw evidenceに保持した。
pool supervisorは両jobの回収後に正常終了し、所有slotは全てstoppedを確認した。
GPUを確保したままにしない。実験完了時点では追加PR/mergeを行っていない。
その後、ユーザーの依頼によりコードと検証結果をレビュー用Draft PRとして公開する。
mainへの採用・mergeは引き続き保留し、全モデルの高速化を主張しない。

再現は同一sourceからregistryの `GLOBAL_OWNER_WINDOW_RECOMPUTE` と
`GLOBAL_OWNER_WINDOW_COMBINED` を `make_algorithm_registry().dumps_plan(...)` で
別JSONへ保存し、次の既存CLIをそれぞれfresh processで実行する。

```bash
python -m benchmarks.training.cell_mass_mnist \
  --create-radius-initialization output/init.pt --radius-width 1024 --seed 41
python -m benchmarks.training.cell_mass_mnist \
  --initialization output/init.pt --kind cst --data-directory output/mnist-data \
  --radius-plan output/selected-plan.json --allow-radius-1024 \
  --epochs 3 --batch-size 4 --output output/selected-mnist
python -m benchmarks.training.cell_mass_mnist \
  --initialization output/init.pt --kind dense --data-directory output/mnist-data \
  --radius-execution global-response4 --allow-radius-1024 \
  --epochs 3 --batch-size 4 --output output/dense-mnist
```

CPU math libraryのthreadは全て1に固定する。pool用driver原本は各jobのsnapshot、
ローカル準備版は `output/owner-window-followup/mnist_driver.py` に保持する。

## Proレビュー後の保存済み分布の確認

ChatGPT会話 `6ae93fb6-724c-83e8-b191-32a7fbe64536` のPR #126レビューを受け、
追加学習をせず、上記MNIST jobの `radius-initial.json` と `epochs.json` を調べた。
レビューの提案は、同一状態の局所演算・更新比較、同期方式の交差比較、両学習後checkpointを
両Planで動かす交差評価の3つ。新しいkernel最適化を加える前に、原因を分離する方針を採る。

以下は入力側のbank分類。今回の同一軸・共有半径では出力側も同じatom数だった。
全体52,428 atom。L6はT2/K3、L12はT4/K3。他のbankは保存された全時点で0 atom。
これは既存 `radius_statistics` のCPU FP64診断による分類であり、GPUが実際に生成した
owner keyとのbyte照合は行っていない。特に半径のbank境界付近は今後の局所診断で確認する。

| 状態 | 現行 L12 / L6 | 併用 L12 / L6 |
| --- | ---: | ---: |
| 初期 | 52,428 / 0 | 52,428 / 0 |
| epoch1後 | 52,428 / 0 | 52,428 / 0 |
| epoch2後 | 52,427 / 1 | 52,427 / 1 |
| epoch3後 | 52,423 / 5 | 52,421 / 7 |

最終L12割合は現行99.9905%、併用99.9866%。wide bankへの移行は保存時点では見られず、
固定DFTの適用atomが大きく減ったという説明を支持しない。epoch間の一時的なbank移動は
記録していないため、全45,000 stepの経路が常に同じだったとは断定しない。

owner当たり最大atom数は、両Planとも入力205→205、出力255→258（初期→epoch3後）。
全bankの非空owner数は入力が現行256→261／併用256→263、出力が現行256→261／併用256→262。
この記録からは大きなowner集中も認められない。active ownerごとの実行時間やGPU timelineは
未取得であり、分布だけからCPU律速を確定しない。

今後の短い診断では、P/x/dYと全optimizer stateをそろえ、同一Plan同士の再実行を対照に
置いてから、C/raw/DFTによる一回の演算・更新差を分離する。その後、同じ短区間・平均stepで
毎step同期と区間同期を比較する。自由走行した異なる終点同士だけで性能を判断しない。

`checked_owner_compact_keyed_linear` はTorchDispatchMode、Tensor subclass、tracing等で
別scheduleへfallbackする分岐を持つ。計装する際は実際のkernelとmode 2/6を確認し、
目的の経路を変えていないことを記録する。今回の通常MNISTでfallbackしたという証拠ではない。

集計JSONは `output/owner-window-followup/distribution-summary.json` に保全。
この追記は既存ログの分析のみで、新しいGPU確保・性能測定・再学習は行っていない。
PR #126はDraftのまま、mainへの採用も引き続き保留する。
