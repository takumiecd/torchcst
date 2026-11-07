# 多入力中間値のFP32化とAtom勾配の行並列化

[English](multirow-packed-20261107.md) | **日本語**

先のrow8 pilotでは更新時間を22.7%短縮したが、Z/H/GはFP64保存のままだった。
今回は演算の数学・初期化・支持・正規化・radiusをdetachするtask VJP・native Polar更新を
維持する。production defaultや他GPUは変更しない。

ユーザーの補足に従い、Linearのradius/center decode・routing判定・係数・位相・縮約・
moment/response保存をFP32とする。FP64は比較参照であり、新Linearの要件ではない。
共通host設定scalarや変更しないoptimizerは、このLinearの信号計算とは別である。

| Algorithm末尾 | 逆伝播まで保存するもの | Atom勾配の実行 |
| --- | --- | --- |
| packed | FP32 Z/H | 従来の行順次縮約 |
| packed_recompute | X/P/keyのみ。FP32 Z/Hを再計算 | 8行groupとFP32部分縮約 |
| packed_parallel | FP32 Z/H | 8行groupとFP32部分縮約 |

既存8行展開に加え、response projectionも8行共有とする。
parallel版はAtom幾何を小さい一時FP32 tableへ一度計算して、行groupをCTAへ分散する。
R256/A4096でFP32のgeometry/partial scratchは0.390625/1.5 MiB。保存activationではなく一時領域である。
dense Wの構築、Atomの省略、tile別の再正規化はしない。
再計算はforward時のSavedVariablesを使い、version検査・saved Tensor hook・一階微分制限を維持する。

対象は256↔1024 FFN、A4096、FP32、R1〜256、Polar modes8/base1/merge2/spacing1、
radius[.5,50]。4行・FP64 row8の対照は別に選べるままとする。
Tensor/function変換は旧経路へ明示fallbackし、その場合の性能・メモリは主張しない。
Graphは拒否する。

計時前にmetadata・直列化・保存方針、GPU境界・端数、既存global-native回帰を確認する。
dXのみ・Atomのみの逆伝播、逆伝播再実行・変更検知を調べる。
保存するbulk配列がFP32で、再計算版はZ/Hを保存しないことを確認する。
各候補のFFN両方向R256/A4096の全VJPをCPU FP64 Fourier/autograd参照と照合し、
既存atol3e-5/rtol3e-4と独立polynomial求積を維持する。
実4block taskの再開とrow8対照の3更新で勾配・Parameter・Adam moment/clockを
既存許容差で比較する。精度変更後の完全一致は必須とせず、別に報告する。

既存cst-research FFN learnerと同じTinyStories token/data順序を使う。
A10080GBPCIe MIG3g.40gbのみ、TF32off、幅256/4block/4head/FFN4N、
batch2×context128（R256）、accumulation4、seed17、c8/A4096、q0、minimum-001、
AtomLR16倍、baseLR3e-4、AdamW(.9,.95)/decay0/clip1/native finite_chord、
LR/data horizon20M token固定。baseline4・row8・Dense・新3候補をfresh process各20更新でscreenする。
既存の計時sampleは更新11〜20、メモリには20更新全部を含む。
候補選択用であり、採用計測や収束の研究ではない。

本命は再計算しないFP32並列版とし、再計算版は時間・メモリの選択肢として残す。
数値gate合格後に全候補をscreenし、再計算しない候補の最速と再計算版を、
baseline4・row8・Denseとfresh100更新・反転2順序で確認する。
Aはbaseline4→row8→Dense→本命→再計算、Bは再計算→本命→Dense→row8→baseline4。
数値gate不合格の候補は失敗を保全し、計時しない。
既存learnerのmedianは更新11〜100、メモリは100更新全部を対象とする。
準備・外部資格検証/cold compile・評価・checkpoint/probeは別枠。
process全使用量やGraphの主張はしない。
CSTの初期Parameter・分散合わせreportを揃え、loss健全性・source/result hash・全process観測を残す。
同じallocationはハードウェア独立再現ではない。
数値不合格・nonfinite・OOM・時間予算で固定試行を止め、失敗を保持し許容差を緩めない。
raw evidenceはignored outputに保全する。

本書は計時前に固定するprotocolで、GPU結果ではない。
