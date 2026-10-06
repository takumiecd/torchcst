# 2026-10-07 小さい Linear の夜間改善

64/128とも、探索・並べ替え・メタデータの更新と、parameter勾配の仕事分割を改善した。denseへは近づいたが、まだ同等には届いていない。単一の構成が全条件で最速になる結果ではない。

B32、N64/A204、N128/A819、初期 decoded rho 1.25/3/8/mixed（すべて >1）、FP32 IEEE / TF32 off。測定対象は forward/backward/production fused AdamW + Polar 更新を含む complete step。各実行21サンプル、独立2実行、2回目は条件・plan順を反転。以下のセルは実行中央値の中央値。denseは同じshape/input/target/dtype/測定境界の独立初期化 nn.Linearで、同一初期行列や同一更新軌跡ではない。

## 繰り返し確認した測定

### L4: 合成構成 ([PR #44](https://github.com/takumiecd/torchcst/pull/44))

最速の列は測定後に選んだ固定構成の記述であり、自動dispatcherの性能ではない。

|N / 初期rho|copy8 µs|最速測定 µs|dense µs|dense比|copy8比改善|allocated peak B|選択した構成|
|---|---:|---:|---:|---:|---:|---:|---|
|64 / 1.25|54.89|49.88|39.14|1.27x|9.1%|115,712|`local-ordered-reuse-paramatom16-param4-param2-atom32`|
|64 / 3|59.07|51.36|39.30|1.31x|13.1%|115,712|`local-ordered-reuse-paramatom16-param4-param2-atom32`|
|64 / 8|56.44|51.14|39.11|1.31x|9.4%|115,712|`local-ordered-reuse-paramatom16-param4-param2-atom32`|
|64 / mixed|62.55|56.15|39.21|1.43x|10.2%|99,328|`local-ordered-reuse-paramatom16-param4-param2-split2-atom32`|
|128 / 1.25|65.96|60.76|45.75|1.33x|7.9%|307,200|`local-ordered-cache-repair8-param2-copy8-atom32`|
|128 / 3|70.13|66.12|45.59|1.45x|5.7%|307,200|`local-ordered-cache-repair8-param2-copy8-atom32`|
|128 / 8|73.86|66.18|45.49|1.46x|10.4%|307,200|`local-ordered-cache-repair4-param2-copy8-atom32`|
|128 / mixed|75.95|69.31|45.37|1.53x|8.7%|307,200|`local-ordered-cache-repair4-param2-copy8-atom32`|

### L4: 64向け cached atom16 ([PR #47](https://github.com/takumiecd/torchcst/pull/47))

最速の列は測定後に選んだ固定構成の記述であり、自動dispatcherの性能ではない。

|N / 初期rho|copy8 µs|最速測定 µs|dense µs|dense比|copy8比改善|allocated peak B|選択した構成|
|---|---:|---:|---:|---:|---:|---:|---|
|64 / 1.25|55.46|49.36|39.39|1.25x|11.0%|117,248|`local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32`|
|64 / 3|59.49|51.16|39.32|1.30x|14.0%|117,248|`local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32`|
|64 / 8|57.31|50.33|39.26|1.28x|12.2%|117,248|`local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32`|
|64 / mixed|63.24|56.58|39.33|1.44x|10.5%|117,248|`local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32`|

### G4: middle/mixed ([PR #44](https://github.com/takumiecd/torchcst/pull/44))

最速の列は測定後に選んだ固定構成の記述であり、自動dispatcherの性能ではない。

|N / 初期rho|copy8 µs|最速測定 µs|dense µs|dense比|copy8比改善|allocated peak B|選択した構成|
|---|---:|---:|---:|---:|---:|---:|---|
|128 / 3|64.17|60.46|43.26|1.40x|5.8%|307,200|`local-ordered-cache-repair8-param2-copy8-atom32`|
|128 / mixed|70.17|64.27|43.23|1.49x|8.4%|307,200|`local-ordered-cache-repair4-param2-copy8-atom32`|
|64 / 3|55.59|50.20|33.72|1.49x|9.7%|115,712|`local-ordered-reuse-paramatom16-param4-param2-atom32`|
|64 / mixed|58.59|54.53|33.70|1.62x|6.9%|115,712|`local-ordered-reuse-paramatom16-param4-param2-atom32`|

## 数学と比較の境界

位置勾配、Y、dX、すべてのsource parameter勾配を独立FP64 oracleで照合。20 stepのcaptured optimizer/reference比較で更新値・moments・stepを確認。既存契約のdecoded幅固定VJPを保持し、forwardの幅はoptimizerで変化する。全domainの離散L2正規化をslice前に行う。singleton判定はrhoだけではなく、実際の全domain支持点が1個でnormがfloor以上であることを確認する。

同じGPU/条件内のCST構成・独立実行では初期Parameter bytesを一致確認し、denseも含めてinput/target bytesを一致確認。L4とG4の初期CST Parameter bytesは異なるため、GPU間の純粋なハードウェア倍率は主張しない。各GPU内の比較が主結果。source/result archiveとcommitted/submitted/worker runtime hashを保存した。

allocated peakはcapture/replayとlive buffersを含む。CST reserved peakは6,291,456 B、dense reserved peakは48,234,496 B。dense allocated約34 MBにはworkspaceが含まれ、parameter保存量ではない。Hの新しいglobal配列は導入していないが、DRAM通信ゼロやL2常駐は測っていない。compiler spill/shared値からキャッシュ常駐を断言しない。

## 有効だった方向と負の結果

- 13 fieldのphysical metadata copyを並列化すると128が改善。
- parameter VJPをbatch方向に2分割すると64/128が改善し、64では16 atom / 4 warpの分割が有効。
- exact cached orderingは現在のgeometryで再検証し、近傍repairで直せない場合full sortへ戻す。128/rho3ではrepair8が52 refresh中49/50回のfull sortを1/1回へ減らす。ただしsort回数だけで最速は決まらない。mixedではrepair4が安い場合がある。
- 64のcached atom16ではnarrow gather、wide repair4が独立2実行ともuncached対比で改善。middle/mixedの小差は再現方向が揃わない条件がある。
- 診断カウンタを外すとcache allocated peakは512 B減る。速度差は小さいため独立反復で判断。
- prefix maxima + 二分探索はlayoutの追加費用が勝ち、全体で悪化。serial batch loopもparallel split2に届かない。
- histogramとcopyの融合は128で悪化。histogramのpaddingを小さくした構成は64に有望なscreen結果があり、反復結果を別表に保存。
- owner partitionを減らすとpeakを削減できるが、128 wideは遅くなる。64 mixedは小さな改善余地がある。

## 並列性と次の方針

hostの統括は1つ、所有GPUは同時に1つ、実験jobは逐次。subagentは使用していない。GPUのCTAは並列で、batch16はbatch行、owner16は連続した入力/出力次元、consumer atom32はatom recordのchunk。parameter atom16/32はatom recordの分割であり、128次元からランダムに16個選ぶ意味ではない。

次は、128のmiddle/mixedに残る更新費用とparameter VJPを優先する。64は小さいfixed recipeを維持し、narrow/middle/wide別に測る。一般geometryへ拡張する前に支持集合の検証を増やす。metadataのphysical viewで消費しないfieldを省く案はまだ未実装で、将来候補に留める。

実験はkernel/ブランチとdraft PRに保存。新しいdraft PRは自動mergeしていない。08:00 JST以降は新規実験を開始せず、回収・所有runtime停止・証拠保存を完了する。
