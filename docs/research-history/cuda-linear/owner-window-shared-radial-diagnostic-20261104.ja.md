# 8係数共有Aの限定実現性診断（事前仕様）

## 研究費用をかける範囲

Proとの設計レビュー（turn `875aac7d-82ed-44ca-89ca-a94dac171109`）とCPU source監査から、
既存geometryを保ったAだけについて、数値・生成コード・実メモリの限定診断を行う。
速度採用を判断する36短期cellや45,000更新は、このjobから自動開始しない。
baselineは#132 `fba2e72d733325fbd7be5aac334a0a7463d96b27`。#133/#134の判定は維持する。

recapture前には旧programを退役・GC・同期し、証明を更新してから次のGraphを作る。
「保存wが必ず二組生存する」としてAを除外しない。一方、旧Graphを保持したeager評価と
private poolの重なりを、Tensor論理容量だけで合格とも判定しない。

## 一つのproducer/consumer変更

current FP64 geometry、classifier、Rによるbank分類、sort32、Flat C、window、Polar更新を保つ。
forward contractで元のprofile関数を呼び、元のcell-phase recurrenceと式

\[
w_k=\frac{2c_k\,\mathrm{float}(\Delta)\,s_k}
{\mathrm{PI32}\,k\,\mathrm{float}(\Delta)/\mathrm{float}(P)},\quad k=1,\ldots,8
\]

を同じ評価順で作る。既存local c配列のc1..8をwへ置き換え、forward二軸で使い、
元Parameter行に対応するFP32 `[8,A]` をsave_for_backwardで保存する。
backwardはwを読み、profile生成をしない。c0=1/float(P)、q0=float(spacing)*c0を保つ。
中心phase、alias/Nyquist、q/dq/C/H/G/勾配の加算順、FP64局所減算とPolar pullbackを保つ。
reciprocal化、c*scale、Horner、direct sin(k*phase)、TF32、bank-table別案は入れない。

追加保存は1,677,696 bytes/forward。新launch/barrierなし。scalar load/storeを使い、
contiguousだがstorage_offsetが非ゼロの合法なhook payloadも扱う。
新payloadはdevice/FP32/shape/contiguousをreverse sortより前に検査し、Pのversion保護を残す。
dX-onlyでも保存する。backwardが存在しないno_grad/非微分eager呼出しは元の#132経路を使い、
不要なforward→backward保存をしない。この条件分岐は最初のGPU測定前に固定する仕様であり、
評価・capture込みのメモリ計測は省略しない。公開dispatcherは変えない。

## 診断の順序と停止条件

新しい一つのL4 allocationで次の順に実施する。controlのruntimeは未変更#132 archiveを使い、
比較用benchmark instrumentationだけを同じものへ揃える。実際にloadした.so、source/build flags、
PTX/cubinまたはSASS/resourceを保存する。artifact保存用の--keepはコード最適化設定と区別して記録する。

1. #132で8実更新した状態と、#132自身45k checkpointを凍結する。実MLPから得た固定X/dYで、
   全52,428 atomのkeys/geometry/Y/dX/全dPと、同じP/m/v/clockからの1回のnative更新をbyte比較。
   代表rank6/12/17、K2、周期/広い支持、相殺、部分勾配、offload/retained/multiple/current-P、
   新payloadの拒否と非ゼロoffsetの合法restoreも検査する。
2. 各状態16実Graph更新。clock8→24、45000→45016、42状態＋5勾配、metrics/order、
   capture復元、fallback0を一致させる。仮更新は数えない。
3. 数値が通った場合だけ、同じprospective memory-v2.1の8cycle×16実更新を比較する。
   128実更新/120replay、report自身のendpoint、summary>=epoch peak>=endpoint、cycles3–8 plateau、
   allocated増分<=2MiB/reserved<=8MiBを維持する。
4. capture/restore、旧Graph保持評価、退役GC同期、証明更新、再capture、最終退役のcurrentと
   resetしていないpeakを対称に記録する。追加warmup/empty_cache/途中peak resetによる救済はしない。
5. 通常の最適化設定の生成コードで、候補reverseからprofile呼出しが削除されたかを確認する。
   register/stack/local命令をusへ換算せず、旧#133のsector数をこのbinaryへ流用しない。

byte不一致、guard迂回、想定外fallback、memory判定失敗、削除機構の不成立があれば停止して保全する。
診断の短い時間を5%採用の証拠にしない。すべて成立しても、このjobはそこで終了し、結果をProと
レビューしてから正式な短期性能研究を行う価値があるか判断する。

正式研究へ進む場合も、両状態>=5%/各6勝/byte一致/memory-v2.1を保ち、すべて通った場合だけ
2順序の45kへ進む。この事前仕様や#134の不合格を、結果を見て緩めない。

## 準備と結果の区別

本記録の初版時点ではGPU診断は未実施。ローカルCPU検証は1739 passed/2439 skipped。
対象CPUは23 passed/44 skippedで、CUDA検証の代替ではない。host C++ syntax、Ruff、宣言検査はpass。
