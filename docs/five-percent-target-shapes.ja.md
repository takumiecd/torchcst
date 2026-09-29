# 5% CST学習の対象形状

候補リストのGPU予約と完全ステップCUDA Graphを含む後続の検証は[ブレイクスルー探索](five-percent-breakthrough-research.ja.md)を参照。
2026-09-29の小形状・Ampere再測定は[こちら](small-shape-ada-ampere-20260929.ja.md)を参照。

2026-09-28に初版、2026-09-29に優先順位を更新。atom数を`round(0.05*N*N)`に固定し、重み幅を変える。Mは線形層へ渡す入力の総行数であり、系列モデルのmicrobatch sizeそのものではない。

## 優先順位

| 形状 | 役割 |
| --- | --- |
| 1024²、M=16/128/2048 | 小形状の必須性能目標。小さいMでの固定費、denseとの速度比、ピーク割当を同時に評価する。 |
| 8192²、M=2048 | 大形状の必須性能目標。完全なAdamWステップの速度とdenseより低いピークを同時に評価する。 |
| 8192²、M=128 | 小さいMでの速度・メモリ回帰を確認する。 |
| 4096²、M=128/2048 | 新しい最適化が8192だけに特化していないか確かめる。 |

1024²のFP32 dense Wは4,194,304バイト、5% CSTのatomパラメータは1,048,580バイト。8192²では順に268,435,456／67,108,860バイト。4096²から1024²へ下げるとW由来の絶対的な節約量も16分の1になる。5%はatom数の割合であり、生成されるWの非ゼロ率ではない。

## RTX 6000 Adaでの1024²実測

seed 21、単層AdamW、FP32、TF32無効、全出力の許容`atol=rtol=3e-5`。CSTは512行の局所W窓を使い、forwardからbackwardへのW保持窓は0枚。学習中に確保するW窓の合計容量も全W未満とする。全W・全dWはCST学習経路に作らない。denseとの速度は同一GPU・同一プロセス内で10回ずつ交互に測り、メモリは各方式を別プロセスで測ったPyTorch割当ピーク。denseの初期W生成とCSTの正しさ照合用全W生成は、ステップ時間・ピーク測定に含めない。

| M | dense時間中央値 | CST時間中央値 | 対応ペアのCST/dense比中央値 | denseピーク | CSTピーク | ピーク削減 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.256 ms | 1.355 ms | 5.31倍 | 40.29 MB | 30.08 MB | 25.3% |
| 2048 | 0.521 ms | 1.698 ms | 3.25倍 | 68.21 MB | 62.37 MB | 8.6% |

両方とも全出力照合は違反0。denseのM=128は各回0.230–0.365 msと短時間測定の揺れが大きく、保持窓1枚の別runでは4.07倍だった。速度比は概ね4–5倍の目安。M=2048ではactivation等がピークを支配し、1024²のCSTによる節約は約5.8 MBしか残らない。大形状のメモリ利益と小形状の速度差は別々の必須指標として評価する。

再現コードは`prototypes.profile_paired_dense_cst`と`prototypes.benchmark_mapped_training_memory`。測定JSONは`output/ada-20260928/cache-hypotheses/paired-dense-cst-1024-m128-cache0.json`、`paired-dense-cst-1024-m2048-cache0.json`、`step-1024-m{128,2048}-cst-cache0.json`、`step-1024-m{128,2048}-dense.json`。旧A100測定との比較は[サイズ別メモリ記録](five-percent-mapped-training-memory.ja.md)を参照。

## 1024²向け経路とdispatchの判断

現行の1024²学習は8192²と同じlisted候補、局所W生成、局所dWからatom勾配への集約を使い、W窓とGEMM精度だけサイズに合わせる。`torch.profiler`下の1ステップGPU kernel時間はM=128で0.848 ms、M=2048で1.182 ms。いずれも局所W生成4回が約0.260 ms、atom勾配2回が約0.212 msを占める。M=2048のGEMM群も合計約0.371 ms。profiler時間は通常の同期wall時間とは異なる。GEMMだけを別kernelへ切り替えても、1024²の差全体は解消しにくい。

forward専用の既存融合候補をM=128で比較した。局所W＋GEMMが0.180 ms、A100向け融合設定が0.174 msだったが、後者は4 MiBのscratchを要求し、1024²全Wの4 MiBと同容量。scratchなしの単純融合は0.507 msで遅い。両候補は出力照合に通ったが、backwardは未実装であり、1024²学習用の代替backendとして採用できない。

`CSTLinear`には既に`_IMPLEMENTATIONS`のbackend登録と`auto`選択がある。ただし現行`auto`は入力のM、GPU、学習時のメモリ上限を見ず、単一chartでは全Wを作る`materialized`を選ぶ。mapped試作の学習経路もこの登録表の外にある。**新しいPyTorch dispatcherを先に設けても、今の1024²を速くするkernelは増えない。** 次の実装順は、全W相当の一時領域を避ける1024²用forward/backward経路を試作・検証し、実測で勝った時にshape・M・dtype・device・学習モードを受ける小さな実行ポリシーへ登録すること。明示的なbackend指定を残し、自動選択が全Wへ戻らない条件をテストする。

内訳とforward比較のJSONは`output/ada-20260928/cache-hypotheses/kernels-1024-m128.json`、`kernels-1024-m2048.json`、`forward-alternatives-1024-m128.json`。計測コードは`prototypes.profile_mapped_training_kernels`と`prototypes.profile_ada_forward_alternatives`。

## dispatch前の1024²計算方式比較

同じRTX 6000 Ada、5% atom、FP32、全W未満の局所窓で、AdamWステップを交互測定した。下表の時間は各実験内の中央値であり、実験間ではGPU・ホスト負荷による変動がある。局所W窓を小さくする案は、forwardで保持するWを増やしてbackwardでの再生成を減らす場合も試した。括弧内は`(窓行数, forwardから保持する窓数)`。

| 方式 | M=128 | M=2048 | 判断 |
| --- | ---: | ---: | --- |
| listed、(512, 0) | 1.610 ms | 1.648 ms | 各実験の基準 |
| listed、(256, 1) | 1.789 ms | 2.069 ms | 遅い |
| listed、(256, 2) | 1.734 ms | 2.015 ms | 遅い |
| listed、(128, 4) | 2.539 ms | 2.811 ms | 遅い |

小さい窓ではピークがM=128で最大約1 MB下がるが、W生成とGEMMの呼び出し回数が増え、速度の利益はない。M=2048の窓変更によるatom勾配の相対L2差は最大約`5.1e-7`。行列積の分割順でFP32の丸めが変わるため、個別要素に設定した`3e-4 + 3e-4*|基準|`を最大23個超えたが、forwardと入力勾配は同じ要素別許容内だった。

atomを2個並列化し4 warpで512行窓を生成するkernelは、交互GPU event測定で`0.0753→0.0726 ms`。全Wを作らず、生成Wの最大要素差は`4.7e-9`、出力違反0。学習ステップへ組み込んだ再測定ではM=128が`1.490→1.466 ms`、M=2048が`1.672→1.642 ms`で、ピークは変わらなかった。約2%の小改善なので試作モード`listed_parallel`として残し、自動選択にはまだ入れない。M=128の最初の測定には外部負荷と見られる大きな時間変動があり、上記は空き状態での再測定値。

atom勾配の別方式を同じlisted forward/入力勾配経路で比較した。M=128では基準`staged_listed`が`1.483 ms`、`atom_major`が`1.509 ms`、`staged`が`1.736 ms`、`fused`が`1.862 ms`。M=2048では順に`1.671/1.789/2.021/2.126 ms`。`interval`は両方約3 ms。別方式のatom勾配は基準に対する相対L2差が約`5e-6`で、要素別許容を数百箇所超えた。速度も精度も基準を上回らず、採用しない。

Tensor Coreを使う既存のFP16x3 GEMMも1024²で試した。forwardと入力勾配をFP16x3、局所dWとatom勾配をIEEE FP32にする。M=2048では、これと2-atom並列W生成の併用が最も速かった。同じプロセスでdense、従来CST、新CSTを24回ずつ交互に測り直した結果は以下。各CST方式は局所W窓512行、保持0枚で、ピーク割当も同じだった。

| M | dense | 従来CST | FP16x3＋並列W生成 | 新/従来の対応ペア比 | 新CST/denseの対応ペア比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.311 ms | 1.460 ms | 1.471 ms | 1.015倍 | 4.77倍 |
| 2048 | 0.513 ms | 1.633 ms | 1.459 ms | 0.897倍 | 2.86倍 |

M=2048では従来CST比約10.3%改善。新経路とdenseの全出力照合は`atol=rtol=3e-5`で違反0、最大絶対差`8.5e-7`。従来CSTとの比較でもforward、入力勾配、atom勾配の確認に通った。M=128は負荷の変動が大きく、対応ペアでは新経路が約1.5%遅い。小Mへの採用は見送る。**1024²でもMによって有利な計算方式が変わる**ことが実測できたが、denseとの速度差はM=2048でも約2.86倍残る。dispatch実装はまだ行わず、この2経路を試作のまま保持する。

再現コードは`prototypes.profile_small_window_schedule`、`prototypes.profile_parallel_weight_atoms`、`prototypes.profile_paired_dense_cst`。JSONは`output/ada-20260928/cache-hypotheses/`の`small-window-schedule-m*.json`、`parallel-weight-atoms-1024-m128.json`、`parallel-weight-step-1024-m*.json`、`atom-comparison-1024-m*.json`、`gemm-comparison-1024-m*.json`、`combined-comparison-1024-m*.json`、`paired-dense-cst-1024-m*-threeway.json`。次はW生成とatom勾配にまたがる融合案を小形状で試し、Mの切替境界を測定してからdispatchへ載せる。
