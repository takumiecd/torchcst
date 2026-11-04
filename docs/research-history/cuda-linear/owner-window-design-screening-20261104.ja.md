# 次の実装より先に行った費用モデルの検討

ユーザーの「むやみに実装を繰り返すより熟考を優先する」という補足を受け、
#134の結果から直ちに次のprototypeへ進む方針を改めた。#132をbaselineとして維持し、
新しいruntime実装・Plan・GPU jobは追加していない。周期内wrapを省く未使用helperの草案は
ignored patchに保管してworking treeを元に戻した。研究の継続を、実装数の増加とは扱わない。

## raw producerの不要処理と、回収できる費用は別である

Proは、wide windowが必要とするraw依存だけを残し、level別gridへ狭める案を提案した。
コード上の不要処理の指摘は妥当だが、まず保全済みのkernel名別集計とtraceをCPUで再集計した。
新しいGPU測定ではない。

| #132 baselineの状態 | raw生成 us/update | window生成 us/update | 未計装全体の5%に相当する参考値 us/update |
| --- | ---: | ---: | ---: |
| 初期 | 27.754375 | 17.318063 | 29.018343 |
| #132学習後 | 27.698250 | 20.080063 | 29.477546 |

従来のprocessed JSONにある`moments_kernel`は両者を含んでいた。
各16更新のraw eventは288件、18/update。全eventのgridは543 blocksだった。
固定host順からlevelを対応させると、level2のforward/reverseは各約3.54/3.52us、
残り16 launchは各約1.27–1.32us。level名はtraceの直接metadataではなく順序からの推定である。

提案は18 launchを残すため、回収可能な費用はraw生成全体より小さい可能性がある。
ただし計装時間を未計装時間の厳密な上限とせず、この表だけで5%改善不能とは断定しない。
「消費されないrawがある」という正しさの証拠と、「今ここへ実装費用をかける価値が高い」
という判断は区別する。Proへ分離結果を送り、実装前の再検討を依頼した。

## owner pairごとの密行列化にも、組み立ての費用がある

rank r=12のatom aについて、入力windowからの応答と出力windowへの寄与を係数
v_a、u_a、振幅s_aで表す。異なるowner pair gへatomが分かれるとき、

\[
M_g=\sum_{a\in g}s_a u_a v_a^T
\]

を毎更新で作る設計を考える。これは現在Pからの再構築を省略しない概念比較であり、
実装済み経路ではない。forwardの積和項数だけを理想化すると、

\[
F_{\mathrm{direct}}=2rAB,\qquad
F_{\mathrm{pair}}=r^2 A+r^2 GB.
\]

ここでGは空でないowner pair数。profile生成、ampの置き場所、変換、packing、同期と
backwardを含まないので、これを速度予測や実GPU FLOP数とは呼ばない。
単なるFMA項数の比較では、pair方式が小さくなるには

\[
B(2A-rG)>rA
\]

が必要で、Gが大きければ大batchだけでこの不等式を満たすこともできない。

保全済みの#132由来＋256更新後checkpointをCPUでdecode/classifyすると、次となった。
入力中心はp[:,3]、出力中心はp[:,2]で、native式に従った。CPUの超越関数をCUDAの
再実行とは扱わず、GPUの全key配列は保存されていないことを明示する。

| 状態 | rank12 atom数 | owner pair数G | atom/pair平均 | 最大atom/pair | pair/direct項数比 | FP32 pair行列を全保存した場合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初期由来＋256更新 | 52,428 | 36,276 | 1.445253 | 6 | 5.651522 | 19.927002MiB |
| 45k由来＋256更新 | 52,423 | 36,261 | 1.445713 | 6 | 5.650201 | 19.918762MiB |

この二つのsnapshotでは、同一pairで多数atomを集約する利益は小さい。
全行列を保存するなら、現行の追加allocated予算2MiBも大きく超える。tilingで保存を減らす場合は、
代わりにpacking、再読込、再計算とbackwardの費用をモデルへ戻す必要がある。
約5.65倍という値も、ハードウェアの利用率差を考慮しない項数比であり、実際に5.65倍遅いとは言わない。

## Tensor Coreのピーク値を、そのままCSTの利益としない

[NVIDIA Ada whitepaper Appendix D](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
のL4仕様はFP32 30.3TFLOPS、TF32 dense60/sparse120TFLOPS。疎性のない行列化へ
sparseの120を適用しない。[CUTLASSの3xTF32例](https://github.com/NVIDIA/cutlass/blob/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu)
はFP32を複数のTF32積へ分解する。追加の積・分解とB4のpadding、毎更新の係数準備、
全atom勾配までを含める。ピーク値や項数だけで性能を肯定・否定しない。

現在のbyte互換性を守る変換と、数学的には同値でも縮約順を変える行列化は別の設計空間である。
[PyTorchの数値精度の説明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)でも、
数学的同値だけではbitwise一致を保証しない。後者を研究するなら数値契約を前向きに定める必要があり、
今回のbyte gateや過去の不合格を事後に緩める理由にはしない。

## 次の設計を実装する前の問い

1. 同じ値・同じ仕事をforward/backwardで何回作っており、何回へ減らせるか。
2. 保存・global traffic・register/stack・同期をどれだけ増やすか。
3. 残る費用を考慮しても、完全stepに意味のある利益が残る根拠があるか。
4. 数値・hooks/live settings・メモリ契約を保てるか。何が分かれば案を捨てるか。

半径由来の係数の共有もこの観点で検討中だが、新候補へ決定していない。
実装数を増やす前に、こうした費用と反証条件を揃える。

[処理済み再集計とCPU費用モデル](owner-window-design-screening-results-20261104.json)に値と
元checkpointのSHAを保管した。元のGPU source/result/manifestと#134の見送り判定は変更しない。
raw trace、CPU分析プログラム、検討草案はignored `output/bank-reuse/` に保存している。

## Proの再検討と、係数共有の一回の静的監査

Proはraw分離結果を受け、raw producerを第一候補とする推奨を撤回した。
raw producerとTensor Core化を実装せず、半径/bankから決まる8個の積分済み係数を
共有する案について、まず生成コードを精査する提案となった。こちらの設計草案とも一致する。

現ソースではprofile生成が2A回、cell係数生成が4A組ある。8値を共有すれば、設計上は
各A回へ減らせる。ただし同一kernel内のcompiler共通化は別途確認が必要であり、
ソース上の回数を実命令削減数としない。

今回の純粋#132/#134実測binaryは保存されていないため、保全済み#133の通常sm89 buildで
PhaseShared=falseのbaselineをCPUから読んだ。これは当該buildに限定した静的監査である。
profileの定数・HarmonicPhase・profile_values_m8部分のsource SHAは#132と一致するが、
係数関数のwrapperとcallerは#133で変わっており、現在のbinaryとの同一性を証明しない。

- 各方向のcontractには、R/P、PI乗算、cosine/power/polynomialと9出力に対応すると
  推定されるcalleeへの静的CALLが一箇所ある。chunk内のatomごとに実行する構造である。
- 対応calleeの静的spanはforward455命令、reverse453命令。lineinfoや内部calleeの
  symbol labelがなく、値の流れからの対応付けである。動的命令数やusへ変換しない。
- 通常のpolynomial部分には9個のFP32出力STLがあり、LDLはなかった。source上の
  power配列が、その部分でlocalから繰り返し読み込まれるという説明は支持されない。
- calleeの他の範囲には特殊関数の遅い経路もある。全静的local命令や、過去の全local
  transactionを、この共有で回収できる費用とは扱わない。
- cell/wの二軸間共通化と、wを新しく保存した場合の正確な丸め境界までは解決できていない。
  cの出力がFP32であることだけで、積分後wのbyte互換性を証明したことにしない。

保存側の採算も厳しい。8個のFP32係数を追加すると1寿命あたり1.60MiBであり、
仮に二つが重なれば3.20MiBとなる。これはピーク実測ではないが、2MiB guideを
満たすと説明するには不十分である。

別の表現として、既存2×AのFP64 geometryを「8係数＋FP32振幅」の9×A FP32へ
置き換える草案も考えた。論理増分は1寿命1,048,560 bytes、二寿命で2MiBより32 bytes小さい。
512-byte丸めの単純モデルでは増分がちょうど2MiBになる。しかし既存Rの全consumer確認、
classifier側へ移す仕事とresource増加、saved-state契約、allocator pool/recapture/reservedを
解決していない。これも実装・メモリ合格の証拠ではなく、別のproducer/consumer設計案である。

**静的監査は重複profileの存在を支持するが、5%を超える純利益、積分後wのbyte互換性、
capture込みメモリ予算の成立は確認できていない。係数共有もruntime実装・GPU実験へ進めない。**
未解決点を、楽観的なcache仮定や計装値の引き算で埋めない。

[静的監査と保存量のモデル](owner-window-coefficient-sharing-audit-20261104.json)に
binary/sourceの由来、対応箇所、静的命令と限界を記録した。
