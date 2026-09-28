# 5% CST学習の対象形状

2026-09-28。atom数を`round(0.05*N*N)`に固定し、重み幅を変える。Mは線形層へ渡す入力の総行数であり、系列モデルのmicrobatch sizeそのものではない。

## 優先順位

| 形状 | 役割 |
| --- | --- |
| 8192²、M=2048 | 主性能目標。完全なAdamWステップの速度とdenseより低いピークを同時に評価する。 |
| 8192²、M=128 | 小さいMでの速度・メモリ回帰を確認する。 |
| 4096²、M=128/2048 | 新しい最適化が8192だけに特化していないか確かめる。 |
| 1024²、M=128/2048 | 固定費とactivationのためCSTの利益が縮む境界ケース。主な最適化目標にはしない。 |

1024²のFP32 dense Wは4,194,304バイト、5% CSTのatomパラメータは1,048,580バイト。8192²では順に268,435,456／67,108,860バイト。4096²から1024²へ下げるとW由来の絶対的な節約量も16分の1になる。5%はatom数の割合であり、生成されるWの非ゼロ率ではない。

## RTX 6000 Adaでの1024²実測

seed 21、単層AdamW、FP32、TF32無効、全出力の許容`atol=rtol=3e-5`。CSTは512行の局所W窓を使い、forwardからbackwardへのW保持窓は0枚。学習中に確保するW窓の合計容量も全W未満とする。全W・全dWはCST学習経路に作らない。denseとの速度は同一GPU・同一プロセス内で10回ずつ交互に測り、メモリは各方式を別プロセスで測ったPyTorch割当ピーク。denseの初期W生成とCSTの正しさ照合用全W生成は、ステップ時間・ピーク測定に含めない。

| M | dense時間中央値 | CST時間中央値 | 対応ペアのCST/dense比中央値 | denseピーク | CSTピーク | ピーク削減 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.256 ms | 1.355 ms | 5.31倍 | 40.29 MB | 30.08 MB | 25.3% |
| 2048 | 0.521 ms | 1.698 ms | 3.25倍 | 68.21 MB | 62.37 MB | 8.6% |

両方とも全出力照合は違反0。denseのM=128は各回0.230–0.365 msと短時間測定の揺れが大きく、保持窓1枚の別runでは4.07倍だった。速度比は概ね4–5倍の目安。M=2048ではactivation等がピークを支配し、1024²のCSTによる節約は約5.8 MBしか残らない。1024²を主目標にすると、大形状で重要なメモリ上の利益と計算密度を評価しにくい。

再現コードは`prototypes.profile_paired_dense_cst`と`prototypes.benchmark_mapped_training_memory`。測定JSONは`output/ada-20260928/cache-hypotheses/paired-dense-cst-1024-m128-cache0.json`、`paired-dense-cst-1024-m2048-cache0.json`、`step-1024-m{128,2048}-cst-cache0.json`、`step-1024-m{128,2048}-dense.json`。旧A100測定との比較は[サイズ別メモリ記録](five-percent-mapped-training-memory.ja.md)を参照。

## 1024²向け経路とdispatchの判断

現行の1024²学習は8192²と同じlisted候補、局所W生成、局所dWからatom勾配への集約を使い、W窓とGEMM精度だけサイズに合わせる。`torch.profiler`下の1ステップGPU kernel時間はM=128で0.848 ms、M=2048で1.182 ms。いずれも局所W生成4回が約0.260 ms、atom勾配2回が約0.212 msを占める。M=2048のGEMM群も合計約0.371 ms。profiler時間は通常の同期wall時間とは異なる。GEMMだけを別kernelへ切り替えても、1024²の差全体は解消しにくい。

forward専用の既存融合候補をM=128で比較した。局所W＋GEMMが0.180 ms、A100向け融合設定が0.174 msだったが、後者は4 MiBのscratchを要求し、1024²全Wの4 MiBと同容量。scratchなしの単純融合は0.507 msで遅い。両候補は出力照合に通ったが、backwardは未実装であり、1024²学習用の代替backendとして採用できない。

`CSTLinear`には既に`_IMPLEMENTATIONS`のbackend登録と`auto`選択がある。ただし現行`auto`は入力のM、GPU、学習時のメモリ上限を見ず、単一chartでは全Wを作る`materialized`を選ぶ。mapped試作の学習経路もこの登録表の外にある。**新しいPyTorch dispatcherを先に設けても、今の1024²を速くするkernelは増えない。** 次の実装順は、全W相当の一時領域を避ける1024²用forward/backward経路を試作・検証し、実測で勝った時にshape・M・dtype・device・学習モードを受ける小さな実行ポリシーへ登録すること。明示的なbackend指定を残し、自動選択が全Wへ戻らない条件をテストする。

内訳とforward比較のJSONは`output/ada-20260928/cache-hypotheses/kernels-1024-m128.json`、`kernels-1024-m2048.json`、`forward-alternatives-1024-m128.json`。計測コードは`prototypes.profile_mapped_training_kernels`と`prototypes.profile_ada_forward_alternatives`。
