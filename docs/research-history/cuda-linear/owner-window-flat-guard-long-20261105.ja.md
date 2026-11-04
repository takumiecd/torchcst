# G1: 二順序45,000更新の事前仕様

実行条件は、Stage3最終追加試行の全Gate通過と原本回収・CPU直接検証の完了である。
この文書の作成時点ではStage3が実行中であり、長期実験の実施報告ではない。
候補の核心4ファイル、src242、学習runner・数値body・計時loop・自然なlifecycleを変更しない。

## 費用・順序・更新数

共有poolのL4同時1台、新しい1 job、hard timeout1200秒、保存用90秒reserve。
初回process最大500秒、以降最大180秒、全体残り時間で上限を縮める。再試行・追加trialなし。
import/build/capture/保存等を含む既存wallログで実現可能性を確認し、明らかに不足なら起動しない。

最大8 fresh process。job内extension/Triton cacheは共有し、fresh processとcold cacheを区別する。

| 順序 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| A | CST qualified | CST flat | dense flat | dense qualified |
| B | dense qualified | dense flat | CST flat | CST qualified |

BはAの全順序を反転する。各CST pairが揃い次第判定し、不成立なら後続processを起動しない。
各cellは同じ凍結初期値・seed41・全train60,000/test10,000・真B4から新しいtrajectoryを開始する。
3epoch×15,000＝45,000実更新で、最初の8 ordinary更新も含む。Graph replayは44,992回。
仮capture更新は復元し実更新へ数えない。既存45k checkpointからさらに45k進める実験ではない。

実在する`full_step_graph_learning`へ`--mode graph --flat-c --guard qualified|flat --epochs 3`を渡す。
`--steps-per-epoch`はdefault0。learningでは最初の8更新もFlatCであり、Stage3 comparisonの
Sort32 bootstrap→FlatCとは別に記録する。N1024/A52428、FP32 body＋既存FP64 geometry/Polar、
TF32off、math threads1、同じoptimizer則・支持・全勾配/dX・hooks/live契約を保つ。

## 状態と実行来歴

全3epochのPT/JSONを保存し、CST42状態＋5勾配、dense20状態＋5勾配、totals、train/test metrics、
order、初期値/data/optimizer契約を比較する。kind内qualified/flatとA/Bの全trajectoryをbyte一致させる。
開始時の`initial_state`はlazy optimizerを含むため、epoch末の42/20件やclockを初期snapshotへ強制しない。
epoch末clockは15,000/30,000/45,000。公開exact clock validatorでCST6/Dense5保存pathと値/bytesを照合する。
凍結済み#132 long参照との適用範囲とhashを開始前に固定し、原PTとの直接比較は回収後CPUでも確認する。

native資格・flat有効性・Plan/recipe/entry・実loaded executor .so path/SHAを全captureで確認する。
CSTは自然な8/15000/30000でのcaptureを期待し、denseは適格なら既存Graphを再利用する。
同kindのcapture/retire位置・理由を一致させ、強制recaptureを追加しない。fallback0、finite、
captureのstate/metrics復元、評価によるstate/gradient不変を既存runnerどおり要求する。

## 時間・memory・採否

主時計は既存reportの二つを両方使う。

\[
T_{train}=\sum_{e=1}^3 T_{train,e},\qquad
T_{adopt}=T_{preparation}+T_{train}.
\]

trainingは全45k更新のgather/loss/backward/実optimizer/metrics/guard/index/version通知を含む。
preparationにはoptimizer作成、capture/復元/資格、自然な再prepare、退役/GC/syncが既に入る。
capture/retirementの個別時間を再加算しない。evaluation・checkpoint/provenance I/Oは除外する。
import/data/model配置、初期evaluation、epoch shuffle/reset等も既存時計外であり、process全wallとは呼ばない。
初期evaluation内のcold native compile/loadは正確な独立timerがないため未分離と明示する。
process wall/build/cache証拠は別に保存し、残差をcompile時間と断定しない。計時のためにrunnerを変更しない。

**各順序のCSTについてtraining、adoptionの両方で`1-T_flat/T_qualified >= 0.05`を要求する。**
順序間平均で一方の未達を救済しない。5%未満も停止し、都合のよい順序・seed・反復を追加しない。
同値/性能未達と、数値・来歴・memory不成立や未完了を区別して記録する。
denseにも同じ実行基盤と正誤/memory条件を与え、速度は共通基盤効果の対照として全て報告する。
denseの改善でCSTの未達を救済しない。

各順序・kindでcapture込み累積peakのflat増分をallocated≤2MiB/reserved≤8MiBとする。
optimizer前の既存reset以降にpeak reset/empty_cacheを追加しない。全更新・評価・capture/復元・退役を含む
既存scopeとepoch endpoint `after_eval_train_sync_before_checkpoint_and_graph_retirement`を維持する。
Stage2のCST公式memory-v2.1/dense手動継続条件は継承し、3epochを8cycleへ加工しない。

原本回収・直接再検証後にのみ採用可否を判定する。比較は資格補強済G0対G1であり、
元#132からの純高速化や、dispatcher既定への採用を意味しない。mainへのmergeはこの実験に含めない。

最終GPU driverはCPUで実mainの8セル全経路と、性能未達・schema/clock/backend/memory不正・timeout/保存失敗時の
停止/保存を予行してから投入する。synthetic fixtureは数値・性能の証拠には数えない。

## 実施結果と回収検証

source `02881f03bbc5557e23d09d8a2fd2308d966ad3ee`、job `l4job-85f5659c0a384c16b4205d804efdfaea`。
L4/Torch2.11.0+cu130/CUDA13/Triton3.6、native backendで最大8セルを全て完了した。
各45,000実更新、合計360,000実更新。driverの保存込みwallは549.0秒。所有VMは停止・回収済み。
これは一allocation内の8 fresh processであり、8台の独立GPU再現ではない。

| 順序 | CST学習 qualified→G1 秒 | 短縮 | preparation込み qualified→G1 秒 | 短縮 |
| --- | ---: | ---: | ---: | ---: |
| A | 37.339295→29.977973 | 19.715% | 38.423431→31.071188 | 19.135% |
| B（全逆順） | 34.478560→29.931161 | 13.189% | 35.554074→31.008194 | 12.786% |

両順序でtraining/adoptionの5%基準を通過した。順序差を隠す平均や再測定は使わず、両値を残す。
denseも同じguard変更によりtraining7.677%/7.032%、adoption7.290%/6.617%短縮した。
共通基盤の改善であり、差分の百分率をCST固有の効果やCUDA演算量削減とは扱わない。

全4pairのcapture込みallocated/reserved増分は0/0 bytes。
CSTのpeakは261,020,160/465,567,744 bytes、denseは273,735,168/452,984,832 bytes。
allocated/reservedの順であり、GPU process全usageの計測ではない。
CSTは8/15000/30000、denseは8/15000の自然captureで、qualified/flat・A/B間のeventsも一致した。
denseをCSTに合わせて強制再captureしたものではない。

全3epochのCST42状態＋5勾配、dense20状態＋5勾配、totals/metrics/orderは、pair間・順序間・凍結済み#132参照と一致した。
最終testはCST loss0.083672397/accuracy97.37%、dense loss0.070017517/accuracy97.87%で、同kindの全条件で同じ。
clockは各epoch末15000/30000/45000、native資格・実Plan/entry/.so・capture復元・fallback0も全て通過した。
これは保存したepoch末の直接検査であり、全45,000時点のPTを保存したという意味ではない。

回収CPU検証はraw441/441、元SHA/sizeへ復元した24 PT、source303/runtime242/核心4、7入力・先行Gate receipt、
全8cell・4pair・歴史参照との24epoch直接比較を通過した。`integrity_passed`/`verification_complete`/`stage4_pass`は
全てtrue、欠落・未検証・errorは0。source/result/driver/検証receiptのSHAと全8セルの二時計は同名JSONへ記録した。
数値body・runtime・learning loopは変更していない。最終的な研究opt-inとしての採否をProへ提示し、
公開dispatcherや未変更#132に対する純高速化と混同せず判断する。
