# Quadratic B-spline Hの連続配置とendpoint集約

2026-10-19。前の[係数前計算・H分離比較](quadratic-bspline-h-precompute-20261019.ja.md)を基準に、同じ数学で読み取り配置と書き込み方法を比較する。
研究用revision v13。公開dispatcher・KernelSpec・optimizer・task VJPの幅stop-gradientは変更しない。

## 数式と保持する精度

生profileを `f_i(a), f_o(a)` とすると、

\[
H_{ab}=\sum_i f_i(a,i)X_{bi},\quad
G_{ab}=\sum_j f_o(a,j)dY_{bj},\quad
D_a=\max(\sqrt{Q_{o,a}Q_{i,a}},\epsilon),\qquad
Y_{bj}=\sum_a (A_a/D_a)f_o(a,j)H_{ab}.
\]

`Q`は全観測siteのprofile二乗和。degree-4の離散momentによる既存の正確なL2と中心微分を使い、積分近似・軸別floor・支持の打ち切りを追加しない。
FP64の準備・係数・累積値・加算、既存のFP32 H/G丸め、FP32 X/P/Y/dX/dPを保つ。
幅はatomごとに異なり、Polar更新でstepごとに変わる。task勾配で幅を直接微分する実験ではない。

quadratic profileは4個の切替位置と3個の多項式係数で、

\[
H_{ab}=-\sum_{r=0}^{3}\sum_{k=0}^{2}c_{i,ark}P_{bk}(t_{i,ar}),\qquad
\Delta_{bkt_{o,ar}}\mathrel{+}= (A_a/D_a)H_{ab}c_{o,ark}
\]

と計算する。差分配列のscanと多項式評価でYへ復元する。backwardもG、H/Gの中心微分、正規化微分とdXの同じ構造を使う。
係数とcutは毎forwardで現在の幅・中心から作り、backwardにはそのforward時点のsnapshotを渡す。

back queryが計算する縮約は、中心微分を添字μで表すと、

\[
S_a=\sum_b H_{ab}G_{ab},\quad
T_{o,a}=\sum_b H_{ab}G_{\mu_o,ab},\quad
T_{i,a}=\sum_b H_{\mu_i,ab}G_{ab},
\]
\[
dA_a=S_a/D_a,\qquad
d\mu_{d,a}=(A_a/D_a)\bigl(T_{d,a}-S_a\,\partial_{\mu_d}\log D_a\bigr).
\]

floorが有効なときはその枝の分母微分を0にする。振幅のtask勾配は既存Polarの2成分へ戻す。
HとHμ、GとGμはそれぞれ同じmoment loadを使うが、別の係数積・FP64演算が必要になる。
forwardのHを保存しないためbackwardは入力bankとHを再計算する。これらとbatch方向の縮約をback queryの時間に含む。

## 比較する実行方法

- `h-row-split`: `[plane,batch,site]` bankと差分配列、H/Gを `[atom,batch]` に保存する分離対照。
- `h-point-split/fused`: bankと差分配列を `[plane,site,batch]` にし、batch laneのaddressを連続にする。変換・復元のtransposeも完全stepに含む。fusedではH/Gをglobal保存せず、その場で注入する。
- `h-input-sort/share`: 入力4cutをlexicographicに毎forwardでsort。8atom内で同じcutへのbank loadを一つにし、GPU内gatherで共有する。係数やH自体の共通化ではない。nativeの幅順は保ち、元のatom順へのdP書き戻しを合成する。
- `h-local-reduce`: 隣接する同じcutの寄与をblock内でscanして、最後のlaneだけatomic加算する。非連続に再登場する同じcutも正しく扱うsegment-start flagを使う。
- `h-site-owner/owner-share`: 4A endpoint eventを整数histogram/scanで振り分け、cut 1..N-1を担当するCTAが全寄与を集約して通常storeする。cut 0の偏りは固定容量の部分和と別reductionで処理する。広いσのY/dX注入に浮動小数点atomicを使わない。native Wとevent配置の整数atomicは残る。site×atom支持表は作らないが、endpoint用のindex配列は追加する。

H/Gの全体workspaceを持つのはsplit/owner候補だけで、forward内の最後の利用後に解放する。backwardまでHは保存せずXから再計算する。
空の支持、振幅ゼロ、片側のみの勾配、retained backward、Graph内のσ切替を検査する。振幅ゼロでも振幅勾配を捨てない。

## 測定条件

L4一台、1024×1024、A=52,428、spacing .1、FP32、TF32 off。
AdamW lr=1e-4/wd=.01/fused/capturableと既存fused Polar更新。warmup 5、21 replayの中央値。
準備・routing・sort・係数・正確なL2・全勾配・optimizer更新を含む未計装完全stepと、Graph capture/replay込みallocated/reserved peakを比較する。
stage診断は別captureで、stage時間の和は完全stepの代用にしない。

初期mixed条件のCPU構造診断では、broad 41,942atom、入力4cut tuple 10,328種類。
入力cut順8atomではbankの論理的なendpoint参照が167,768→39,252（76.6%減）になる。
これは初期配置の要求数であり、DRAM転送量・cache hit率・live全stepの削減率ではない。

## 探索結果

初回B32 mixed実測（job `l4job-183d090f787847c3a7eb11cea26b0c61`、source `a538501f`）：

| 方法 | 完全step ms | allocated / reserved MiB |
|---|---:|---:|
| 既存H cut .2 | 2.570 | 68.488 / 132 |
| row split | 2.856 | 73.941 / 172 |
| point split | 2.463 | 74.643 / 174 |
| point fused | 1.983 | 68.488 / 134 |
| input sort | 2.080 | 68.488 / 134 |
| input share + sort | 2.182 | 68.488 / 134 |
| local reduce + share + sort | 2.596 | 68.488 / 134 |
| site owner | 4.349 | 78.366 / 174 |
| owner + shared reads | 4.575 | 78.365 / 158 |
| W range reuse | 3.222 | 78.590 / 154 |
| Dense | .0829 | 49.002 / 106 |

分離したrow→point対照ではH query .362→.175ms、Y inject .543→.441ms、back query .736→.717ms、dX inject .546→.437ms。
back queryはG/G'とH/H'の読み取り、FP64演算、batch方向のatom勾配縮約を含む。
fusedのforward .780→.454ms、backward 1.145→.859msも短縮した。

明示的な共有は要求数を減らしてもsort約.133msとGPU内部gatherを必要とし、完全stepでは不採用。
shared-read backwardのregister数は126→167、shared memoryは1→2KiBへ増加したがspillは0。
ownerはfloat atomicを消しても各siteのeventループ、H/Gの再読込、部分和reductionとevent構築が必要で、injectは約1.33msずつに増えた。
これは今回の実行方法の負の結果であり、すべての出力担当方式が遅いことの証明ではない。
ハードウェアcache/atomic stall counterは未測定で、物理DRAM帯域やatomic競合の割合は断定しない。

最終比較と検証の記録は以下へ追記する。

参考のlaunch探索（job `l4job-9c80730e83ec410abf798bbf28f216d7`、source `329de44e`）では、atom tile 4/8/16、warps 1/2/4を比較した。
完全stepは1.982～2.115msで、基本の8atom/4warpsを超える改善はなかった。
cutoff .05（小さいσも広いσ側に振り分ける）は2.152ms、cutoff .5は2.369ms。
compactな3係数planeでは2.086ms、allocatedは62.888MiBへ減るがreservedは142MiBだった。
このlaunch探索はdriver成功・archive取得後のremote cleanupで接続を失い、poolがVM停止とserver assignment消失を確認した。正式な最終比較には含めず、負の探索記録として保持する。
精度を緩めず両方のメモリ指標でDense以下にするという採用条件は、ここまでの候補では満たしていない。

## 回帰検証

- `tests/test_quadratic_bspline_cuda.py`: **485 passed**（6宣言/metadata + 479実CUDA）、624.87s。新規は1宣言 + 40実CUDA。job `l4job-095b17925f0d4244896778acbce9cec9`、source `329de44e`。
- CPU全体: **1973 passed / 3255 skipped**、18件の既存warning、41.97s。skipはGPU/DB検証完了を意味しない。
- 新方式の独立FP64 gateはY/dX/全Polar4成分のmaxabsとrelative-L2を両方 `<4e-4` とし、閾値を緩めていない。floor有効/無効、64/1024 site、strided X、各stepの新しいX、live幅・中心変更、Graph内切替、retained forward、ゼロ振幅、dXのみ/dPのみも検査した。
- 完全step測定とstage診断は既存runnerを使い、CST候補間の初期P、およびDenseを含む全候補のX/targetの一致、全atomの幅更新を結果JSONで確認する。
- wheel/sdistの343 runtimeファイルを現checkoutとbyte照合。package SHAはignored evidenceの`package/proof.json`に保存する。

GPU suiteには既存経路のAccumulateGrad stream mismatch warningが1件あり、今回新設した警告ではない。検査は全件完了した。
