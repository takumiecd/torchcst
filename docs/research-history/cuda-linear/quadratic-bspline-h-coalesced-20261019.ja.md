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
- `h-site-owner/owner-share`: 4A endpoint eventを整数histogram/scanで振り分け、cut 1..N-1を担当するCTAが全寄与を集約して通常storeする。cut 0の偏りは固定容量の部分和と別reductionで処理する。自作のendpoint注入kernelは浮動小数点atomicを使わず、cut 0の最終合算にはTorch reductionを使う。native Wとevent配置の整数atomicは残る。site×atom支持表は作らないが、endpoint用のindex配列は追加する。

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

## 8atom/4warps版の完全step比較

独立した2つのdriver processで各条件を測定した。同じL4 VMを排他的に順番に使い、各processは21 replayの中央値を出す。表の範囲は2つの中央値のmin～maxであり、42独立runの分散ではない。

| 条件 | 方法 | 完全step ms | allocated / reserved MiB |
|---|---|---:|---:|
| b64-mixed | range-reuse-a16w1 | 3.2588–3.2590 | 78.840 / 156 |
| b64-mixed | h-cache-edges | 4.4087–4.4202 | 71.191 / 132 |
| b64-mixed | h-point-fused | 3.2913–3.2966 | 72.693 / 132 |
| b64-mixed | h-point-compact | 3.4835–3.4902 | 67.092 / 140 |
| b64-mixed | h-point-c05 | 3.3876–3.3884 | 72.693 / 132 |
| b64-mixed | dense_linear | 0.1070–0.1074 | 49.502 / 106 |
| mixed | range-reuse-a16w1 | 3.2339–3.2341 | 78.590 / 154 |
| mixed | h-cache-edges-c02 | 2.5764–2.6371 | 68.488 / 132 |
| mixed | h-point-fused | 1.9876–1.9882 | 68.488 / 134 |
| mixed | h-point-compact | 2.0906–2.0974 | 62.888 / 142 |
| mixed | dense_linear | 0.0823–0.0829 | 49.002 / 106 |
| small | range-reuse-a16w1 | 3.1839–3.1840 | 78.590 / 154 |
| small | h-cache-edges | 1.4707–1.4726 | 68.488 / 132 |
| small | h-point-fused | 1.4843–1.4865 | 68.488 / 134 |
| small | h-point-compact | 1.4836–1.4869 | 62.888 / 142 |
| small | matrix-support4a16 | 1.2080–1.2082 | 50.231 / 114 |
| small | dense_linear | 0.0824–0.0829 | 49.002 / 106 |
| switch | range-reuse-a16w1 | 3.2313–3.2333 | 78.590 / 154 |
| switch | h-cache-edges-c02 | 2.6319–2.6428 | 68.488 / 132 |
| switch | h-point-fused | 1.9096–1.9107 | 68.488 / 134 |
| switch | h-point-compact | 2.0131–2.0139 | 62.888 / 142 |
| switch | dense_linear | 0.0823–0.0831 | 49.002 / 106 |

混合B32は旧Hから23～25%短縮し、Wから約39%短縮した。Denseに対しては約24倍が残る。この8atom設定のB64は旧Hから約25%短縮するが、Wが約1%速く、最速方式のDenseとの差は約30倍のままである。
B64の旧Hはcutoff .5、新Hは .2。幅の振り分けの選択も比較に含む。同じcutoff .5の新Hも3.388ms前後まで短縮しており、配置変更単独の比較も改善している。
小さいσだけではHを使わない `matrix-support4a16` が1.208msで最速。今回のH変更を小σ専用条件の改善とは扱わない。
メモリはB32の新Hがallocated同等、reserved +2MiB。B64の新Hは旧Hよりallocated +1.50MiBだが、Wに対してはallocated約6.15MiB、reserved24MiB少ない。Denseの両peak以下という採用条件は満たさない。

B32混合のstage診断では、準備（L2を含む）約.162ms、係数前計算.060ms、広いσのH→Y融合.454ms、広いσのbackward融合.859ms。狭いσ側のW構築.057msとatom VJP .110msも別に含まれる。stage間の加算から完全stepを再構成しない。

全CST候補の初期P hash、およびDenseを含む全候補の初期X/target hashは一致。すべてのCST候補で最終σが初期から変化したatomは52,428/52,428。
独立FP64 gateのH系全候補・2process・3状態・4条件にわたる最大値は、Y maxabs 2.868e-6 / rel-L2 1.709e-7、dX 3.051e-6 / 1.719e-7、全atom dP 3.363e-5 / 1.460e-7。閾値は両指標とも4e-4未満を維持した。

B64の8atom/4warps版はforward 130、backward **255 register/thread**、shared 2KiB、spill 0だった。B32のbackward 126 register/threadから増加している。これは実コンパイラの報告であり、occupancy/stallの実測値ではない。B64専用にatom tileを4へ縮小する追加比較を行う。

省メモリ探索 job `l4job-3695ad5e2ca442e395324af201bd6bdf` もdriver成功・archive取得後のremote cleanupで接続を失った。poolによるVM停止とserver assignment消失を確認し、この回は最終比較から除外。同じ数式・候補を新規jobへ明示再提出した。除外結果は`excluded-memory.json`と元archiveに保持する。

## B64のregister診断に基づく追加比較

job `l4job-fd4efbf2e08f4e1d8d40e6f0e824834e`、source `84f83d6e`。新しいL4 VMで同じW/Denseも測り直した。以下は各候補1 processの探索結果で、約1%の差を新たな採用勝利とは扱わない。

| 候補 | 完全step ms | backward register / spill slots |
|---|---:|---:|
| point 8atom / 4warps | 3.2878 | 255 / 0 |
| point 4atom / 1warp | 3.2887 | 255 / 44 |
| point 8atom / 2warps | 3.3389 | 255 / 136 |
| point 16atom / 4warps | 3.3689 | 255 / 134 |
| point 4atom / 4warps | 3.2587 | 126 / 0 |
| point 4atom / 2warps | 3.2512 | 255 / 10 |
| compact 4atom / 4warps | 3.4709 | 126 / 0 |
| W range reuse | 3.2475 | — |
| Dense | .1030 | — |

4atom/4warpsでregisterを半減しても、backward spanは1.722→1.698ms、forwardは.884→.909msで、完全stepの改善は約0.9%。register不足だけを減らす変更では大幅な改善にならなかった。全候補のY/dX/全atom FP64 gateは通過し、この追加比較のdP maxabsも4.572e-5未満（閾値4e-4）だった。
allocated/reserved peakはfull係数候補72.693/132MiB、compact67.092/140MiBで同じ。spill slotsはコンパイラの別指標で、Torch allocated peakからその容量を推定しない。

## 省メモリ比較と判断

再提出 job `l4job-c4d2e76319d74c628b2dbf29f2c6a0c2`、source `84f83d6e` はdriver成功・archive/manifest照合・remote cleanup確認まで完了した。B32混合、各候補1 processの追加比較：

| 方法 | 完全step ms | allocated / reserved MiB |
|---|---:|---:|
| point fused | 1.9806 | 68.488 / 134 |
| compact係数 | 2.0839 | 62.888 / 142 |
| native Wをbackwardで再構築 | 2.0373 | 64.488 / 134 |
| compact + native W再構築 | 2.1405 | 58.888 / 142 |
| compact 4atom/1warp | 2.0872 | 62.888 / 142 |
| Dense | .0823 | 49.002 / 106 |

Wの再構築は同じforward時点のmetadataを使い、dXの後にWの作業領域をdWへ再利用する。L2の再近似は行わない。係数縮小は5→3 planeとし、中心微分係数を同じquadraticの関係式から復元する。allocatedは削減できるがreservedは下がらず、両peakでDense以下という採用条件は未達。この追加比較でもY/dX/全atom gateは通った。

今回確認できた主な改善はbatch方向の連続配置とH/G生成直後の融合集約である。入力区間をsortして明示共有する方法、local scanでatomic数を減らす方法、endpoint担当が通常storeする方法は、今回の実装では追加費用を回収できなかった。論理的な要求数やatomic数だけを減らせば高速化する、とは結論しない。

B32混合は約1.99msでDenseの約24倍、小σ専用では既存group版約1.21msが優位。B64はWと新Hが約3.25～3.30msの同帯で、Denseとの差は約30倍。数式や勾配精度を変えずCST内で改善した研究候補として保持し、公開dispatcherへの採用・Dense同等という主張は行わない。

## 証拠と再実行

L4、driver 580.82.07、Python 3.13.15、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。主比較の2 processは同じL4 VM、追加B64/メモリ比較は別L4 VMで対照も測定し直した。

ignored evidenceはworktree内 `output/bspline-h-coalesced-20261019/` に保持する。`verified-jobs/` にsource/result archive、manifest、receipt、driver結果を保存し、`source-byte-proof.json` で全343 runtime・対象tests・凍結したbenchmark実装/Case/Planを照合した。旧studyのrunnerと現runnerの差は追加の診断metadataだけで、主比較以降のrunnerはbyte一致。catalogの追加は許容するが、測定済みPlanの宣言を変更していない。

接続失敗を伴った2回の探索は主比較へ混ぜず、別の除外記録・archiveを保存した。すべての正の性能主張は回収・manifest確認・cleanup確認済みjobに限定する。提出JSONの整合性検査も全件通った。raw log/benchmarkをruntime packageへ移していない。

GPU使用は所有者環境では共有poolを通す。driver中での再実行コマンド例：

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-h-coalesced-final.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit COMMITTED_SOURCE \
  --output "$CST_JOB_OUTPUT/mixed.json"
python -m benchmarks.submissions check "$CST_JOB_OUTPUT/mixed.json"
```

B64/switch/smallは対応する`*-h-coalesced-final.json`、メモリ交換は`mixed-h-coalesced-memory.json`、追加register比較は`b64-mixed-h-coalesced-tuning.json`で再現する。

| job | source archive SHA256 | result archive SHA256 |
|---|---|---|
| `l4job-183d090f787847c3a7eb11cea26b0c61` | `2df6a03cf7710174702cd7dd9bbf76bb366b8df5a887f1dfaefd4efa39e9e53d` | `0514b20eb6efb540c489e0e79ff25eec7e509a2c9f8e93bd4541cd846fb0678e` |
| `l4job-095b17925f0d4244896778acbce9cec9` | `1a23d6893c5df0c5b50007e7806b04b9d4ddfb95c73ed3e3ee63c4a5bdc2b42d` | `af46013abc8f10cc3540428dd8c88c67da76372671ffbe282b60c491ffb1df91` |
| `l4job-f34d4be43254478ab08f8420da2a35ea` | `8317e9b959643c7075f0f72758f633a830aeac5cce85fddead68e934d0216889` | `b37be603567b0dc6e71388a5065b45b32e2489d98a46242d93ee5f684f535c5a` |
| `l4job-2979f0d54c844f4a98f65d4f50be289a` | `a6e6057dee6e0633d0f1278627b8e9fa7d493dd7ad9d8d6f7e8133ab9003ea87` | `200df8f4337bd4091dd83cd8185f44f801331091df92f9408663b77472ca47d9` |
| `l4job-fd4efbf2e08f4e1d8d40e6f0e824834e` | `4debe517693b64efe6e5f28b0d6f0a2303515dc58c2f906122550092fb38d01c` | `f06d2c6ec76e5893d7c02a277ea90cf4b1a2134e86bc5704efe360b102cc95ce` |
| `l4job-c4d2e76319d74c628b2dbf29f2c6a0c2` | `60be3dc69cc3b128d32ec1e28098aefbde97e884712b7926d3c5d17e4cfb40c7` | `6d62562b36b632b1bec0eaaa12b3d19059e704a90a281a5882477027da7e8f71` |
