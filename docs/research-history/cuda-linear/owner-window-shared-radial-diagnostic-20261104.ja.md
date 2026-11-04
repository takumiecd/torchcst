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

## 限定診断の結果

**Aの数値・生成コード・実メモリに関する限定診断は成立した。速度採用は未判定。**
[Draft PR #135](https://github.com/takumiecd/torchcst/pull/135)の測定sourceは
`5429f527f5817072a15e505998ac963ee19d7ba0`。詳細と証拠のhashは
[処理済みJSON](owner-window-shared-radial-diagnostic-results-20261104.json)に記録した。

一回のL4 job `l4job-c6b39d81429a4b16a0301322bc6ae7e5`で、未変更#132 runtime archiveと
候補を別build dirから実行した。controlのsrc全242ファイルを元hashと照合し、benchmarkの
計装だけ対称に揃えた。GPU UUIDは`d2c9ed43-aa0d-7aa0-0c17-f2bf469ebede`、driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6/Python3.13.16。N1024/A52428/真B4/seed41/TF32off/CPU threads1。

- #132の8更新状態と自身の45k checkpointで、同じ実モデルX/dYから得るkeys/geometry/Y/dX/全dPがbyte一致。
  同じP/m/v/clockからsite-only native更新を1回実行し、42状態一致（8→9 / 45000→45001）。
- 各状態16実Graph更新で42状態・5勾配・metrics/order一致、fallback0、capture復元pass。
  保存Tensorのstepを取得後に直接読み、24と45016を確認した。
- 代表rank6/12/17、K2、wide/周期境界/相殺、部分勾配、saved hooks/offload/retained/current-P、
  不正metadata拒否、合法nonzero offset、Graph guard等の対象suiteがL4上で45件pass（44件GPU契約＋1件宣言検査）。
  FP64 Fourier参照による小fixture検証を含む。新たな全52k quadrature検証ではない。
- 8cycle×16実更新（128実更新/120 replay）の各47状態・勾配とmetrics/orderが一致。
  24個のPTを原file bytesと同一に復元し、全642 Tensor比較がbyte一致した。

実際にloadした通常buildの.soとPTX/SASS/resourceを保存した。対応するPTX entryのprofile呼出しは
controlがforward/reverse各1、Aはforward1/reverse0。reverseのregisterは86→72、stackは304→272 bytes。
forwardは79 registers/288 bytesのまま。これは削除機構の証拠であり、時間短縮率ではない。
最適化は元のO2/ftz=false、NVCC --keepは生成物保存用。fast-math/fmad=false/lineinfo/register capなし。

| capture・評価・再準備を含むメモリ | #132 | A | 差 |
| --- | ---: | ---: | ---: |
| peak allocated | 251.9921875 MiB | 251.9921875 MiB | 0 |
| peak reserved | 444 MiB | 444 MiB | 0 |
| 評価後endpoint allocated（全8cycle） | 257,328,128 bytes | 同左 | 0 |
| endpoint reserved（cycle1 / cycle2以降） | 463,470,592 / 465,567,744 bytes | 同左 | 0 |

prospective memory-v2.1の全項目がpass。各31phaseのcurrent/peak allocated/reservedも全て一致した。
追加wの論理保存量・読み書きは存在する。equal peakだけから「無コスト」「allocator再利用を証明」とは
言わない。現在の実測範囲では既存のピークを増やさなかった、という観測である。
Graphの新Plan→recipe→native entryのsource routeは確認したが、このjobでdynamic kernel traceは取っていない。

source archive SHA256は`e781045fe4e9845445333c7dd8e8bb457735ffc027995c9685a10c570bf0a543`、
result archiveは`aa2e7e427c46fcc5e4d35059ba0f73f46f18862c49c1a958a0c268868f61b919`。
raw manifest全1318ファイルを取得後に再hash照合、driver return0/timeoutfalse、owned VM停止を確認した。
原本は共有poolのjob directory、作業用コピーはこのworktreeの`output/shared-radial/results/`に保持。
raw logs/PT/実binary/中間生成物はcommitしていない。

source CI [41818495957](https://github.com/takumiecd/torchcst/actions/runs/41818495957)は
Linux CPU1738 passed/2441 skipped、宣言・wheel/sdist buildを含めsuccess。
ローカルmacOSとplatform skipの内訳は異なる。host syntax最終検査もpass。

正式36短期cell・二順序45kは未実施。16更新の時間や計装付きlifecycle時間を性能判定に使わない。
限定診断結果をProへ戻し、正式短期比較へ費用を払う価値を再レビュー中。#132 baseline、#133/#134の
負結果、Draft/未マージ、公開dispatcher未変更を維持する。
