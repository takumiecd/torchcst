# 8192²アンカーCSTのL4測定とdispatch引き継ぎ（2026-09-29）

## 条件

Colab Pro NVIDIA L4、PyTorch 2.11.0+cu128、Triton 3.6.0。Wは8192×8192、Triweight atom 3,355,443個（5%）、64×64 block、FP32、seed 21。完全なforward・backward・fused AdamWステップをCUDA Graphで測定した。Mは線形層に入る総行数。現行CSTは1024行局所W窓・4窓保持、`listed_bounded`候補・unroll4・候補builder BA32/1 warp、M=2048のforward/dXにTF32x3、M=128にIEEE FP32を使用。アンカーは各blockの固定siteだけで現行atomのTriweightを毎step再評価し、atom振幅と中心をAdamWで更新する。basisはblock局所で再利用する。W全体とdW全体はアンカー学習ステップ中に作らない。

速度は現行CST・アンカー・生成済み全Wを自由パラメータとするdenseを同一プロセスに置き、Graphを交互再実行した同期wall中央値。M=2048は各24回、M=128は16×32が10回、24×48が8回。正しさ照合用の全W生成は時間とメモリ測定から除いた。denseはパラメータ数と学習可能なモデルが異なり、速度だけの参考値。

| M | 各blockのアンカー数 | 現行CST | アンカーCST | dense | アンカー/現行CST・対応ペア | アンカー/dense・対応ペア |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16×32 | 57.87 ms | **33.54 ms** | 12.62 ms | **0.577** | 2.659 |
| 128 | 24×48 | 60.24 ms | 43.69 ms | 12.66 ms | **0.724** | 3.451 |
| 2048 | 16×32 | 148.81 ms | **46.62 ms** | 78.12 ms | **0.322** | **0.598** |
| 2048 | 24×48 | 149.52 ms | **66.95 ms** | 78.17 ms | **0.455** | **0.868** |

16×32はblock内siteの12.5%、24×48は28.125%でTriweightを評価する。M=2048では16×32が現行CSTの約3.11倍速く、denseより約1.67倍速い。精度寄りの24×48も現行CSTの約2.20倍速く、denseより約1.15倍速い。M=128では両アンカーが現行CSTに勝つ一方、denseより遅い。

## メモリと候補

各方式を**別プロセス**で起動し、warmup後にGraph capture・3 replayしたときのPyTorch最大割当。生成Wで初期化したdenseは計測前にCST layerを解放した。

| M | 現行CST | 16×32 | 24×48 | dense |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 606.14 MiB | **576.98 MiB** | 616.99 MiB | 1076.50 MiB |
| 2048 | 1020.17 MiB | **1020.06 MiB** | 1076.06 MiB | 1376.50 MiB |

M=2048の16×32はdenseより356.45 MiB（25.9%）、24×48は300.44 MiB（21.8%）少ない。8192²の16,384 block×4 row tileでは、アンカー用候補数は平均140.22、最大170で、192枠からのoverflowは0だった。したがってこの測定で全候補spanへfallbackするtileはなかった。atomが更新されるため、候補一覧は毎step再構築している。

## 値・勾配と近似の限界

アンカー上のTriweight値は独立した全W参照と相対L2約2.15×10⁻⁷で一致。現行CSTに対する初期出力・入力勾配・atom勾配の相対L2は、16×32で約0.87%、24×48で約0.45%。これは固定補間演算子と真のCSTの差であり、アンカーkernelの実装誤差ではない。

M=2048で27回固定ランダムMSE更新後、元CSTとの出力差は16×32で**4.60%**、24×48で**3.47%**へ増えた。両者のlossはこの固定ランダム課題では近かったが、実タスクの精度・長期学習・許容可能な出力差は未測定。したがって現時点で近似経路を無条件の`auto`へ入れる根拠はない。

8192²では融合Triton Torus decodeと既存PyTorch decodeの4D座標差が最大0.00293になった。大半径約10,691での三角関数を含むFP32計算順序の差が原因と考えられ、アンカー値が参照から相対L2約0.44%ずれた。融合版でM=128が21.49 msという速い試験は**精度失格のため採用しない**。上表はPyTorch decodeを使い、アンカー値を照合した値。試作コードは256 stationを超える場合に融合decodeを明示的に拒否する。

## dispatchへ渡す条件

- 明示的な近似backendとして開始する。現行atom・Triweightを毎step学習する方式だが、出力と学習軌道は真のCSTと同一ではない。実タスクの品質閾値が決まるまで`auto`選択しない。
- この実測の対象はFP32 CUDA、`BlockStripLinear`のStrip＋intrinsic 4D Torus、Triweight、64×64 block、N/Kが64の倍数、5% atom。L4以外のGPUや異なる密度で同じ速度を仮定しない。
- L4・8192²・M=2048の候補は速度優先16×32と精度優先24×48。M=128も現行CSTより速いがdenseより遅いので、メモリ制約を含む選択にする。
- 固定basisとanchor site位置のみ再利用する。atomパラメータ更新後はアンカー値・候補一覧・support判定を再計算する。候補192枠overflow時は全bucket spanへfallbackするため、dispatch後もoverflow統計と性能回帰を監視する。
- 統合時は同じ課題で数百更新の品質、出力・dX・atom勾配、Graph/eager一致、ピーク割当、L4以外のGPUを検証する。品質閾値を満たさなければ現行CSTを選ぶ。

測定ソースは[`profile_anchor_atom_training.py`](../../../../benchmarks/cuda/linear/profile_anchor_atom_training.py)、[`profile_anchor_large_memory.py`](../../../../benchmarks/cuda/linear/profile_anchor_large_memory.py)、実装は[`anchor_atom_training.py`](../anchor_atom_training.py)。[生JSON](../../../../benchmarks/cuda/linear/evidence/l4-anchor-8192-20260929/)に各replayと誤差を保存した。速度実験のGit archiveはcommit `f71cd01`・SHA256 `63ceff34700ca3cdbb9ca2a9c11421b450c32a250c1c8dda6243ed9e9e2d2199`、24×48メモリ実験はcommit `60970af`・SHA256 `a63354eb08de261a61a784979d6883a915515c88b4f02e10025f395b24038689`。Colab sessionを停止し、active sessionなしを確認した。
