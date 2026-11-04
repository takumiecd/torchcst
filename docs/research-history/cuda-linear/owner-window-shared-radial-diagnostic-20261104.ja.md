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
限定診断結果をProへ戻した。#132 baseline、#133/#134の負結果、Draft/未マージ、公開dispatcher未変更を維持する。

## Pro再レビューと固定Aの正式短期比較（事前仕様）

Proはturn `20e64c39-67a4-49e3-8692-8082a1f0c0c8`（answer `16860010-c879-59ae-a121-dd29afb808df`）で、
限定診断成立・一回の正式短期比較へ進む価値を支持した。必須のCUDA runtime修正はなく、
allocator同値の原因を追う別jobや動的traceの総再取得も不要とした。
代表suiteの一部は`torch.equal`であり、45件すべてが正負ゼロまでのbyte比較という意味ではない。
全atom VJP・保存状態の直接byte照合とは証拠を分ける。

runtimeの`src/`は測定source `5429f527`から変更しない。benchmarkだけに、capture後・計時外の
実recipe/native entry、Graph/eligible/GradMode、実middle/Pのrequires_grad、loaded extensionのpath/hashを
assert・記録するopt-inを追加する。文字列とboolだけを保持し、追加Tensor/Graph参照を残さない。
metadata/hash取得はsetup/throughput/retirementの全時計外。各cellの全47状態・勾配も計時後に保存する。

新しい一回のL4 allocationで、まず追加phase計装なしの8×16 lifecycleを各方式一回実行する。
元の計装付きpassは保存し、同期追加のない実行でもprospective memory-v2.1を通ることを要求する。
compileはここで両方式に等しく済ませる。失敗ならそこで停止する。

通過後のformal cellはfresh process、両状態×6組×(#132/A/dense)の最大36、各256実更新。
CSTには`--bootstrap-fp32 --flat-c-baseline`を両方指定し、finalには#132自身の`fp32_final.pt`を用いる。
denseは同じFP32/TF32off/Graph条件で自身の対応状態を使う。入力・初期値・data orderは従来と同じ。
順序は測定前に以下へ固定し、各方式の各位置が2回、base→A/A→baseが3組ずつになる。

| repeat | 順序 |
| --- | --- |
| 0 | #132 → A → dense |
| 1 | dense → A → #132 |
| 2 | A → dense → #132 |
| 3 | #132 → dense → A |
| 4 | A → #132 → dense |
| 5 | dense → #132 → A |

状態ごとに$q_i=t_A/t_{132}$、$R=1-\exp(\sum_{i=1}^{6}\log(q_i)/6)$。
両状態で$R\ge0.05$かつ全$q_i<1$を要求する。各pairが5%以上という追加gateにはしない。
42状態・5勾配・metrics/order/finite/fallback0、capture復元を維持し、取得したPTも直接比較する。
capture込みallocated差<=2MiB/reserved差<=8MiB、計装なしmemory-v2.1を要求する。

byte/guard/finite/fallback/memory不合格は即停止。有効pairでAが勝たなければ6/6は満たせないため
そこで停止し、残りcellを実行しない。全6勝でも状態のRが5%未満なら、次状態へ進まず停止する。
予定36・実施N・未実施条件と停止理由を残し、有効な負けを捨てて再測定しない。
binaryのdump/圧縮は性能cellの間に挟まず、全比較完了または停止後に行う。

短期throughput、setup、retirement、setup+retirementを分け、最後の合計を二重加算しない。
短期peakはretirement前まで、lifecycleは最終退役までとscopeを区別する。45k償却推計は参考のみ。
このjobは正式短期比較だけで終わり、B/配置variant/新45kを自動実行しない。全gateを通れば、
二順序45kへ進む条件を満たしたことだけを記録する。

### 初回formal試行の通信中断と一回の明示的再提出

job `l4job-2521037625734ba195a3dc801506131e`はColab CLI/Jupyterの
`TimeoutError: Timeout waiting for output`で、result archiveの取得前に中断した。
source archive SHA256は`42fb8ec29e1c7d8159b39b2a9bc4d69312d3cd871ebeeb98254d24cc44b80d23`。
source/spec/transport logを保全し、owned VM停止とpool recoverを確認した。
検証できる結果がなく、性能・数値・メモリの成功/失敗はいずれも不明である。
driverやarchive作成の遅延、runtime消失、通信障害のどれが原因かはローカルログから確定していない。

一回だけ明示的に再提出する。A・入力・順序・速度/数値/メモリgateを変更せず、両方式の
MAX_JOBSを1に抑える。これはcompile時の同時実行数であり、生成コードの算術設定や性能セルを
変えるものではない。driverの実行上限1200秒は維持し、private supervisor wrapperで長いexecの
通信・archive作成の猶予だけ600秒追加する。共有queue・所有権・cleanup規則は変えない。
再び結果取得ができなければ、その状態を保全して停止し、未検証の時間から採用判定しない。

## 正式短期比較の結果：最初の有効pairで停止

**固定Aは現行の採用guideを通らず、採用しない。** 限定診断成立は維持するが、
全体速度の6/6勝利条件は、初期状態の最初の有効pairで満たせなくなった。
[正式比較の処理済みJSON](owner-window-shared-radial-short-results-20261104.json)に詳細を残した。

再提出job `l4job-bd50dc8ab66940c0a6f22e0946901242`はreturn0/timeoutfalseで取得できた。
NVIDIA L4 UUID `19e16bfa-4cca-6304-d11f-180e07a7420c`、driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6/Python3.13.16。提出sourceは`0332eb04836ce83dd9a9b36c7c352da92a7a71d7`、
CUDA runtimeは限定診断の`5429f527`から未変更。MAX_JOBS1はcompileだけの設定で、
性能cellは従来どおりCPU math threads1/真B4/TF32off/Graph、未計装の256実更新。

| 初期8更新後・repeat0 | 完全step ms/更新 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| #132 | 0.5780784023 | 251.5341797 MiB | 442 MiB |
| A | 0.6173719297 | 251.5341797 MiB | 442 MiB |

候補/基準は`1.0679726611210163`、このpairでは候補の所要時間が約6.80%長かった。
6pairの幾何平均を求めた結果ではなく、統計的な一般回帰を確定した意味でもない。
「有効な負けpairを捨てず、その時点で停止する」という事前規則を適用した結果である。
予定最大36に対して実施2。残りの初期CST10・dense6、finalのCST12・dense6、二順序45kは未実施。
denseを測っていないため、このjobからCST/dense速度比を作らない。

両cellで実recipe/native entry、Graph存在・eligible・GradMode・実middle/Pのrequires_gradを
計時外にassert・記録し、基準と候補の入口が正しいことを確認した。loadした実.soを原hashと照合して保存。
dump/圧縮は性能cell後であり、cell間に挟んでいない。両cellは同じ#132凍結start/end/全5勾配・
metrics/order/finite/fallback0を満たし、保存PTの47 Tensorも直接byte一致した。
全Parameterとnative側stepを取得後に読み、共に8→264であることを確認した。

先に行った追加phase同期なしの8×16 lifecycleも、prospective memory-v2.1の全項目pass。
128実更新/120 replay、各47状態・勾配とmetrics/orderが一致し、peakは両者251.9921875/444 MiB。
各8endpoint、cycles3–8 plateau、summary>=epoch peak>=endpoint、予算も成立した。
formal取得分の18 PTをlossless復元し、47×(1 pair+8 epochs)=423 Tensorを直接byte照合した。
これにより、前の計装付きメモリpassだけへ依存していないことも確認できた。

取得した実.so/PTXの内容hashと通常O2/ftz=falseも再確認した。対応PTX entryのbody hashは
限定診断と同じであり、reverseのprofile呼出し削除とREG86→72/STACK304→272は維持されていた。
**削除機構・数値互換・実メモリは成立したが、それだけでは完全stepの利益を保証しなかった。**
追加store/readや依存待ちが相殺した可能性はあるが、この未計装比較だけから原因は確定しない。
原因解明のGPU探索や、B/配置変更による救済へは自動移行しない。

source archive SHA256は`853bd8567fd1d09af52c9a2c6ac16840565bf8c29b867edfd8b9cbe02b02cc6b`、
result archiveは`df359f8db9cd041c102f23a7ed31937e6ff99ffd37bfe4080704287f430abed4`。
raw manifest全764ファイル、候補src全242ファイル、展開した実.so/PTXのhashを再照合した。
原本はpool job directory、作業コピーは`output/shared-radial/formal-results/`に保持し、owned VM停止を確認。
最初の通信中断のsource/logも保持し、その未取得データから成否を推測しない。
再提出の成功は、MAX_JOBSや通信猶予の変更が中断原因を解決したという証明ではない。

#132をbaselineとし、PR #135は負結果を含むDraft・未マージの研究記録として保全する。
