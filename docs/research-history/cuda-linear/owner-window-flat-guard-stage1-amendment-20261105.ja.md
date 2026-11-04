# G1 Stage1: 測定前のrunner不備に対する一回だけの追加試行

amendment ID: `G1-stage1-one-additional-run-20261105`。本書は追加GPU確保前の事前仕様であり、
実験完了や採用を示さない。Proの完了回答と公開source監査を踏まえ、固定G1を判定するために
**初回1回＋追加1回、Stage1を合計最大2試行**とする。3回目は行わない。

## 旧試行は失敗したまま残す

初回job `l4job-6ee82518b27947cb94f8d5b7058875e6` は`incomplete_stage1_harness_failure`。
source `852403bf`、56 tests（40 GPU依存/16 CPU-only）が通過した後、ReadLeaseをunaryとした
検証runnerのassertで終了した。完成数値セル0・費用セル0だが、実験をしていない扱いにはしない。
旧archive、最初の停止理由、source・result SHA、21 raw filesの照合、VM停止は元の記録を維持する。
新jobへ旧途中結果を接合せず、旧56 testsを新jobの実行数へ足さない。

元の「一回限り」を最初から満たした再開ではない。公開sourceから失敗が検証器に局在すると
確認し、実C++ ReadLeaseのCPU回帰4件を通したこと、完成比較から負けを捨てる状況でないことを
根拠に、**新結果を見る前に研究費用の例外を一回だけ追加**する。

## 固定するものと許す差分

- 数式・body・guard・workload・精度・optimizer・hooks/live settings・後続の採用閾値を変えない。
  `src/`全242ファイルは#132 `fba2e72d733325fbd7be5aac334a0a7463d96b27`とbyte一致。
- 四つのguard/coreファイルと旧56 testsの二ファイルは旧source `852403bf`とbyte一致させる。
- 変更はReadLease固定pathの修正、時計外のpreflight/比較条件、既存code/defaultsケースの
  **変更前backendの報告**だけ。pair-costの1024回×5 blockの計時loopを変えない。
- 同じ7入力、初期8/自身45k、CST/dense、qualified/flat、同じentryを使う。
  同一allocation内で新たな8セルを完結させる。

初回private driverはnative-onlyを要求していたが、実backendを報告する数値セルの完成前に停止した。
公開資格仕様は完全なPython predicateへのfallbackを許しているため、このamendmentでは
**両guardで同じbackend（nativeまたはPython）を使うこと**を条件として明示する。
これは未測定段階での比較preconditionの変更であり、暗黙の緩和にはしない。最初の2反例ケース、
数値・費用セルの実backendを照合し、結果はそのbackendに限定する。途中変更・片側だけ異なる
backend、結果を見た後の強制設定や取り直しは認めない。G1のflat証明自体は常に有効でなければならない。

## 一回の上限

| 項目 | 上限 |
| --- | ---: |
| 同時L4 / 追加job | 1 / 1 |
| driver hard timeout | 900秒（初回と同じ、延長なし） |
| 最初のpytest process | 1 |
| その中のケース | ReadLease path4＋既存Linear code/CE defaults2、合計6、skip 0 |
| 数値セル | 2状態×2モデル×2guard＝8、各16更新 |
| paired eligible-cost process | CST/dense各1、合計2 |
| 計時block | guardごと1024 calls×5、既定の交替順 |
| test/比較用process | 最大11（compiler/dependency/archive補助processとは別） |
| 再試行・追加block・追加セル | 0 |
| このjob内のStage2/正式性能/45k | 0 |

最初のCST数値セルは初回compile込みで最大500秒、他セル150秒とするが、各processのtimeoutは
全体900秒の残り時間で制限する。補助dependencyはjob-localのみ。旧56 testsの再実行はしない。
最初の6件には`test_known_live_callable_and_configuration_holes_are_rejected[code/defaults]`を使い、
mutation前の実資格backendをJUnitへ保存する。比較と同じbackendで拒否を確認する。

## preflightと合否の証拠

各数値セルはVJP前に入力・source・選択Plan/recipe・現在参照しているextension moduleの
`__file__`/SHA・runtimeを保存する。backend/flatが構築前なら未取得と明記し、構築後・capture前に
追記する。capture、第1/第16更新、終了でbackend/flatが変わっていないことを確認する。
費用セルも各cohortの準備後と時計の前後にscalarだけのpreflightを保存する。

8 JSONの存在だけでは合格にしない。対応するqualified/flatの保存PTで次を直接byte比較する。

- 同じX/P/dYとY/dX/全dP、開始・第1・第16更新の全状態/5勾配・totals。
- order、初期値・checkpoint SHA、capture復元、clock8→24または45000→45016。
- 実Plan・schedule・entry・実loaded module、両guardのbackend一致とG1 flat有効性。

CSTの開始42状態は初期/自身45kそれぞれの凍結referenceとも照合する。
guard-onlyはstate/grad/version/settings/clock不変、fallback 0、全5 block配列を保存する。
継続screenは既定どおり**CSTの全5 blockでflatが短いこと**。denseは同条件で記録するが、CSTを救済しない。
CST費用screenが未達ならそこで停止し、まだ未実施のdense費用processは起動しない。
二cohort同居の費用セルpeakはdiagnosticで、variant別memoryやmemory-v2.1には流用しない。

source/input/Plan/entry/backend/loaded binary不一致、必要testの失敗/skip、未知wrapper、flat無効、
byte/finite/clock/capture復元不正、正常fallback、guard-only状態変化、費用screen未達のいずれでも停止する。
runtime failure、timeout、回収不能でも追加計算はしない。二度目のharness失敗なら固定G1は
**性能不合格とは呼ばず未判定で保全終了**し、第三試行で救済しない。
同じarchiveの回収・hash再確認と、計算の再実行は区別する。

通過時だけ既定の**8cycle memory-v2.1 → 正式first pair → 最大48セル → 二順序45k**へ進む。
追加trial pairなし、最初の有効CST負けで停止、両状態5%以上かつ6/6勝利、byte一致/fallback0、
allocated＋2MiB/reserved＋8MiBは変更しない。qualified G0の補強費用回収を元#132の純高速化と呼ばない。

## 追加試行前の観測

修正後head `2ce8f3b2`のCPU suiteは、native function/read-lease両opt-inで2026 passed / 2269 skipped。
CPU CI `41880041977`も同headでsuccess。追加6ケースのローカルCPU確認も6 passedで、
code/defaultsの両caseはnative backendを報告した。いずれも対象L4でのamendment実行結果ではない。
Pro回答とprivate driver・raw証拠はignored outputへ保全する。公開結果JSONの初回未完了statusは書き換えない。

## 追加試行の観測: Stage1だけ通過

job `l4job-b48fffb7e8894db7805dce763fc161fc`、source
`920561b4d74724706b23d47bb8a9c28daafd631c`。計画した11 process、6回帰case/skip0、
8数値セル、2費用processを完了した。資格backendは反例case・全cohort・capture/更新終端でnative、
G1のflat証明も有効だった。CSTの実entryは`research_owner_window_fp32_geometry_sort32_flat_c`。

initial8/自身45k、CST/denseそれぞれのqualified/flatについて、保存PTの468 Tensor項目が直接byte一致。
CSTの全52,428 atomのVJP入力/出力・dX/dP、開始/第1/第16更新の全状態/5勾配/totals、
clock8→24/45000→45016、frozen開始42状態、capture復元、通常fallback0を確認した。
10 PTの復元file SHAと1,641 raw manifestファイルをローカルでも照合し、missing/unverified/errorsは0。

eligible-only 1,024 calls×5 blockの中央値は、CSTが631.318→521.870µs、denseが236.223→204.658µs。
両モデルとも全5 blockでflatが短かった。guard-onlyのstate/grad/version/settings/clockとGPU allocator
countersは不変。**これは確認処理の費用screenであり、学習stepの時間、GPU時間、CPU律速の証明ではない。**
qualification補強費用の回収を、元#132より速い証拠とも扱わない。

source archive SHA256は`195966d898081a781023820bbe56f20c6b71b8a30c6167eee2fafab90851ea8d`、
result archiveは`4883f41ab7e39cd01754bfc8d80365a1cdd7e318eadb0be7947849ccd22b707e`、
driverは`45dd33152e0453b3ff7b2b2af68053273502a43703d65db01ab7d34aa8b5fbe0`。
詳細な全5 block配列・機械可読結果は同名JSON、raw原本と再照合はignored outputに保存した。
所有VM停止も確認済み。初回の未完了記録は維持し、Stage1の第三試行は行わない。

現時点ではmemory-v2.1/正式throughput/45k/採用は未実施。
次は[prospective Stage2](owner-window-flat-guard-stage2-20261105.ja.md)だけを検証する。
