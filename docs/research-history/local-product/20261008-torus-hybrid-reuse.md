# Torus profile product: selective saved forward contractions

ユーザーは全中間値をon-chipに置く要件を撤回し、backwardで再利用する価値がある
値をDRAMに保存してよいと指定した（2026-10-08）。精度を通過したPR #78の
bounded-poly版を基準に、保存の費用と再計算の費用を同じcohortで比較する。
単一Strip + intrinsic S1×S2 Torus、全chart L2、一回floor、coupled centre-radius
微分、幅task VJP detachとlive幅更新、現在のAdamW/Torus更新は変更しない。

## 保存内容

Algorithm revision v5（第1版v4）、same launch: 1atom/CTA、site64、4warps。
`save=recompute` は元の数学とFP32縮約順を保持し、Hをbackwardで再計算する。
`save=h` はraw H[B]とq[3]/sinc/nu²/nv²（6scalars/atom）を保存する。
入力側微分用の縮約とnorm微分はbackwardで再計算する。
`save=vjp` はHに加えてH0/H1/H2（入力profileのq各成分に対する微分との縮約）、
5つのnorm微分縮約nua/nu0/nv0/nv1/nv2dをforwardで計算・保存する。
forwardで読み込んだXとprofileを再利用し、backwardの入力微分準備走査を省く。
G/ga/g0はbackwardで計算してdXと全atom勾配に同じCTA内で再利用する。
全A×axis profile/W/dWは保存しない。forwardごとのsource snapshot/cacheは
autograd ctxで所有し、旧forwardやretain backwardで別forwardの値を使わない。

intentional FP32保存容量はh: 4A(B+6)、vjp: 4A(4B+11) bytes。
B32、A52428/209715ではh7.969/31.877MB、vjp29.150/116.602MB（十進）。
これはbufferの容量であり、完全stepの実測peakではない。
compiler register/shared/spill/PTX local load/storeを各方式で保存する。
意図したglobal保存とcompiler spillを区別する。spillがあればその方式のGを
on-chipだと主張せず、数値/完全step/実測memoryで採否を判断する。

## 実行前に固定するcohortと予算

L4一台、B32、N1024/2048、sigma3/8、全atoms約5%、seed41、FP32/TF32なし。
初期p/X/dY、chart、live widths、AdamW/Torus update、全chart norm/floorは同じ。
許容値はmax absolute/relative L2とも4e-4かつelementwise4e-4のまま。
更新比較は既存2e-6、20step momentsは既存2e-5。

correctness: 最大2source版、各900driver秒。3方式の既存78tests（GPU skips不可）、
20Graph steps、partial sites/floor/各gradient要求/empty/旧forward snapshot、
全atomN1024/2048 sigma3/8の3方式Y/dX/全dP/更新を検証する。
48compiler variantsは記録し、値が不明な場合はon-chipを主張しない。
失敗版を保存し、gateを緩めず、同じ失敗jobをretryしない。

性能: correctnessを通過した同じsourceでNごとprimary1job（各1500driver秒）。
各sigmaは既存runnerで最大650秒、isolated process順は
onchip(recompute),cuda-h-saved,cuda-vjp-saved,w-gemm、dense最後。
21samples、未計装完全stepを主結果、capture/replay込みallocated/reserved peak。
phase診断は別Graph。runnerの全atomoracleも改めて要求する。
全route/遅い結果/負の結果を残す。source/result archive hashと実測orderを保存。

保存候補が同じNのsigma3/8両方でrecomputeより3%以上速い、またはallocated5%以上
減かつ時間回帰3%以内なら、そのNだけinverse1job（1500driver秒）を実行する。
逆順はw-gemm,cuda-vjp-saved,cuda-h-saved,onchip（dense最後）、Caseのplansも逆順。
結果に合わせた再計測・追加予算・gate変更はしない。同じ初期fixture/hashと実際の
worker順を確認する。独立確認を通過しなければverified winnerを主張しない。
本stage最大総driver予算7800秒。元の完了済み精度stageの予算は変更しない。
公開dispatcherの既定は変えない。PR #78をこの最終scopeの記述へ更新し、採否を残す。


## 第1版: transport失敗、数値/配置結果なし

source156b9aec4c311ec63df041c06f00b99351b9fb38、
job l4job-23b31ee2d09f4e66b2773ceb2f88f7dd。
source archive25abb9fed85493a136030d66f6f231867d4464d9257ff1266caad5463417169e。
Colab execの接続が失われ、1080秒のCLI上限/1140秒のhost上限で停止した。
POOL_RESULT_READYの応答もresult archiveも得られず、correctness・compiler・driver
実消費は未確認。この版のGPU PASS/FAILや速度を主張しない。資源会計は900driver秒の
枠を消費したものとして扱い、同じjob/sourceをretryしない。
owned session停止とserver assignmentなしを確認。CPU1331pass/2264skip、Ruff、
wheel/sdist build、GitHub CPU CIはPASS（GPU検証ではない）。pool job全体を保全する。

第2版（revision v5）は追加の保存量削減を実装する。
vjp保存版はbackwardがXを読まないため、autograd ctxにXを保持しない。
入力勾配だけを要求する場合はq/normだけを保存し、Hや微分用Hを保存しない。
no_gradの推論は保存bufferを作らない。旧forwardのXを変更してもsaved VJPが
元のforwardの勾配を返すことと、推論値の一致を追加検証する（既存testを緩めない）。
残る第2source版の900driver秒枠で、全方式/全atom/Graph/配置を検証する。
性能cohortの方式・物理条件・事前の採用gate・予算は変更しない。

## 第2版のローカル検査と転送保留

実装checkpoint c18d8877。CPU全suiteは1331pass/2267skip/18warnings
（26.57秒）。GPU対象75testsは未実施。6Plan/4Caseの宣言検査、Ruff、
wheel/sdist buildはPASS。build backendが通常venvにないため、依存の取得を
せず`uv build --offline`で既存cacheを使用した。正しさのcohortは元の78testsに
vjpのX保持不要/旧forwardのX変更/no_gradを扱う3testsを追加した81tests。
数値gate・性能cohort・予算は変更しない。

第1版の凍結archiveから`__pool_driver__.py`を読み直すと、29行目の
compiler audit用`python -c`引数に引用符の構文エラーがあることを確認した。
このdriverはGPU検査を完遂できない。接続切断とこの不備を分けて記録し、
結果が未取得のためkernelの数値失敗や実行時間を推測しない。第2版driverの
引用符を修正し、driver自身と埋め込みPython文字列のAST parseおよびRuffを
ローカルで通過させた。同一失敗版のretryは行わず、第2版の枠を使う。

failed jobのspec/source/remote driver/transport logを
`output/torus-hybrid-reuse/failed-transport-v4/`へ複製し、原本と全hashを照合した。
`manifest.json`、`frozen-driver.py`、`driver-audit.json`も保存。pool statusは
job interrupted、全slot stopped。数値/性能結果は存在しない。

ユーザーよりOpenVLAアップロード継続中との回答を得たため、追加のGPU
ソース転送と結果回収はアップロード完了の連絡まで保留する。帯域使用と
Colab切断の因果は未確認。GPU未検証の第2版はPR #78の研究draftに留める。

Revision v5 gate_driver.py SHA256 `c1f1348116c4ff0aa9ddeba7029b2c5c27c6e1bc97dd9c3815d6b9a8fce363dd`.

Revision v5 performance_driver.py SHA256 `ecf85f8939a9ac57d710ef67530c1c99ade702729ab97a203bbb75aa2601a152`.

## 第2版GPU correctness: PASS

アップロード完了の連絡後、poolのrecoverで旧interrupted jobを整理し、
source71df55ea625ee415bc036201cc4cc5961c9f190fを固定して実行した。
job `l4job-0020d65d07154b26a34a94fbf43c9f50`、237.877driver秒。
L4 / Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0。
81tests PASS（GPU skipなし、75.87秒）、各方式の20Graph stepsとmoments、
全atomN1024/2048 sigma3/8、計12route/caseのY/dX/全dP/更新はPASS。
最大絶対誤差0.000221567438839はN2048 sigma8のH保存dX。全12更新比較はexact。
48compiler variantsはspill0、PTX local load/storeなし。
register範囲はrecompute48–128、h56–146、vjp62–141。
source archive SHA256 `90ade67ea654d2e92a2531ab1842f4e85b257df9c103a901930a09383b9b7744`、
result archive SHA256 `e03022c2c7fde407d8858640570eef9da9d0ecb4911bc95c46424b3fc12c24ec`。
job全体はignored `output/torus-hybrid-reuse/gate-v5/`へ複製/hash照合した。

## 性能記録処理の補修: 比較結果なし、時間予算据え置き

最初のprimary jobsはN1024 `l4job-2db62713aa16484eb1e717bba9598e8b`、
N2048 `l4job-f6160393b0f84b3f8e4d4d37416d607b`。
両方ともsigma3の4方式oracleがPASS後、最初のmeasure/onchipの結果を書き出す
際に`OnchipRecipe.contraction`がないためAttributeError。比較可能な時間/メモリ
結果は得られず、方式の性能勝敗には使用しない。両方のFAIL JSON/raw結果は
`output/torus-hybrid-reuse/metadata-failed-{1024,2048}/`へ保全した。
実消費はN1024 68.298秒、N2048 235.347秒。source archiveは両方
`2a9ba8ea3f372adb1a0975f974bf187251436fe88454af2ba4c569fc8aded445`。
result archiveはN1024 `dc6499867908bb3f358b37b8a6c4d1ae82af987900c9cd7521a4df6e30795e22`、
N2048 `66079f565ba55ae346baddab140aa8b9245a9e1cb2df9cb50c4648091be3cd48`。

runnerの結果metadataだけを修正し、CUDAのsave設定とTorchのcontraction/save_hを
区別する。全6宣言recipeのmetadata回帰テストを追加。CPU1332PASS/2267skip、
Ruff、wheel/sdist buildはPASS。runtime package・Plan・4Caseは精度通過版と
byte-identicalであり、数学やGPU kernelには変更がない。

元のprimary各1500driver秒から失敗jobの実消費を引き、N1024最大1431秒、
N2048最大1264秒を補修版の測定へ使用する。実消費が不明な第1correctness版は
引き続き900秒枠全体を消費として扱う。総7800秒のstage上限を増やさない。
事前の「primary1job/N」からの運用上の逸脱を明記する。比較結果がない記録処理
不具合の補修であり、速度に応じた選び直しではない。有効なprimary比較は各N
一つに固定し、順序・21samples・入力hash・数値gate・採用gate・inverse条件は
変えない。同一失敗source/jobの再実行はせず、補修commitの別jobを使用する。
補修測定も失敗した場合、このstageでさらに補修jobを増やさない。

Metadata-repair performance_driver.py SHA256 `6f6576f6399dd14fa23c26ecd23db870bdb9a1f9fa2c687010ad6185400e16d5`.

## Performance disposition: N1024 confirmed, N2048 incomplete

`cuda-vjp-saved` passed the same improvement gate in primary and independent inverse jobs for both sigma3/8. Complete Graph step reduction is 7.90–7.92% relative to recompute; all full-atom numerical checks and live width updates passed. There are two independent jobs per sigma, each with 21 timing samples. H-only save improved by 0.28–0.61% in observed cases and raised allocated memory; it does not qualify for adoption. It remains a research control.

The VJP-saved route is a validated research option, not a new public default. On N1024 it uses 34.737 MiB peak allocated versus 12.194 MiB for recompute. Torch W+GEMM is still about 2.11–2.17 times faster than VJP-saved but uses 131.120 MiB allocated. Dense remains much faster. The intermediate-reuse gain does not establish large-linear competitiveness.

Uninstrumented complete-step Graph time is the decision metric. Phase medians come from a separate instrumented Graph and are not summed into that metric. Allocated and reserved peaks include capture/replay; total GPU process usage is unmeasured. Values below use ms and binary MiB.

| order | N | sigma | plan | Graph ms | eager ms | allocated MiB | reserved MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| primary | 1024 | 3 | onchip | 292.652815 | 295.395672 | 12.194 | 24.000 |
| primary | 1024 | 3 | cuda-h-saved | 290.953455 | 293.667702 | 14.137 | 64.000 |
| primary | 1024 | 3 | cuda-vjp-saved | 269.533619 | 272.236398 | 34.737 | 116.000 |
| primary | 1024 | 3 | w-gemm | 124.050463 | 231.079909 | 131.120 | 400.000 |
| primary | 1024 | 3 | dense | 0.083014 | 0.657052 | 49.002 | 106.000 |
| primary | 1024 | 8 | onchip | 292.746232 | 295.404735 | 12.194 | 24.000 |
| primary | 1024 | 8 | cuda-h-saved | 290.968041 | 293.666771 | 14.137 | 64.000 |
| primary | 1024 | 8 | cuda-vjp-saved | 269.567167 | 272.252360 | 34.737 | 116.000 |
| primary | 1024 | 8 | w-gemm | 126.570456 | 232.588155 | 131.120 | 400.000 |
| primary | 1024 | 8 | dense | 0.082817 | 0.655797 | 49.002 | 106.000 |
| inverse | 1024 | 3 | w-gemm | 124.747780 | 231.173044 | 131.120 | 400.000 |
| inverse | 1024 | 3 | cuda-vjp-saved | 269.956395 | 272.643027 | 34.737 | 116.000 |
| inverse | 1024 | 3 | cuda-h-saved | 291.416330 | 294.112952 | 14.137 | 64.000 |
| inverse | 1024 | 3 | onchip | 293.125768 | 295.845719 | 12.194 | 24.000 |
| inverse | 1024 | 3 | dense | 0.082926 | 0.658968 | 49.002 | 106.000 |
| inverse | 1024 | 8 | w-gemm | 127.695667 | 233.980986 | 131.120 | 400.000 |
| inverse | 1024 | 8 | cuda-vjp-saved | 269.989919 | 272.760522 | 34.737 | 116.000 |
| inverse | 1024 | 8 | cuda-h-saved | 291.431772 | 294.150487 | 14.137 | 64.000 |
| inverse | 1024 | 8 | onchip | 293.149787 | 295.382238 | 12.194 | 24.000 |
| inverse | 1024 | 8 | dense | 0.082564 | 0.673425 | 49.002 | 106.000 |

N1024 primary job `l4job-aebb8a3edf7d48259bbe6eb50923b1bb` (279.348 measured driver seconds), inverse `l4job-930479e42193468abd3b5c85e6c92b15` (270.134). Both use source commit `8dd36f8e2aef7690cacb02678998ff77d98a3d5e`. All source-file hashes, runtime/physical Case/Plan, initial p/X/target hashes and actual worker orders were verified across jobs. Archive hashes can differ from gzip/container metadata even when every file hash matches.

Primary source/archive SHA256: `9e387718b733442e7b213d9c51d57415aab1dc37163c5f39eda2bc9eeef9a05f` / `9be8eeb548238e55706d6748ea2f8523f05ec16bd70ae0f17478e6af88a1a3b7`.

Inverse source/archive SHA256: `a1d81cdb1312e7ac23c29fe8b8cb4e9a4488e80875da0e99508fa2311aeca6b1` / `7b7bda2b64d2b0301d66138f582ea997f661985830eded6280fda87fb5fb1423`.

N1024 sigma3 separate phase diagnostics: recompute forward-loss 111.355ms / backward 181.077ms; VJP-saved 125.498ms / 143.804ms. Forward work increases while backward saves the input-VJP preparation pass. H-only does not remove that pass. This supports selective reuse but does not identify the remaining hardware bottleneck by profiling.

### N2048 partial evidence

The repaired N2048 primary job `l4job-255301d004834f17b1b2c9dd3a894e0e` reached the declared 650s per-case subprocess limit during sigma3. The outer driver exited after 652.490s with TimeoutExpired. Four full-atom correctness workers and three CUDA measure workers completed; W+GEMM, dense and sigma8 performance did not. The aggregate raw JSON remains RUNNING with seven records and must not be submitted as a complete PASS/FAIL artifact. No inverse job is eligible and no N2048 winner is declared. The stage is not extended or repeated.

| N | sigma | completed worker | Graph ms | allocated MiB | reserved MiB |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2048 | 3 | onchip | 2323.770522 | 48.768 | 148.000 |
| 2048 | 3 | cuda-h-saved | 2317.163663 | 56.656 | 160.000 |
| 2048 | 3 | cuda-vjp-saved | 2147.341153 | 137.056 | 316.000 |

These are descriptive completed workers inside an incomplete cohort. VJP-saved shows 7.59% reduction for this sigma3 run only; it is not an independently confirmed N2048 improvement.

N2048 source/archive SHA256: `8e37f5dc9390e8152270d4b7b409d92c53cfeda0e119975b90ae089d91612da4` / `4a6cffdcb2d21feabe51da8e27118d6bbb72e080805601441223c481be91aab6`.

Raw copies and per-file preservation manifests are in ignored `output/torus-hybrid-reuse/{primary-1024,inverse-1024,primary-2048-incomplete}/`. The one-L4 supervisor drained the queue and exited; all slots are stopped. Failed metadata jobs are retained separately. The next structural optimization must be a new predeclared study; this stage does not repurpose unused inverse budget for another candidate.

## Database provenance and recovery

Ten completed artifacts were stored through the existing PostgreSQL API: four all-atom correctness-only PASS artifacts, four complete N1024 performance PASS artifacts (primary/inverse sigma3/8), and two metadata-failure FAIL artifacts. Exported bytes/SHA256 match every original, and same-provenance reimports inserted no extra run/projection. Provenance contains the owned pool job ID and verified source/result archive hashes; credentials stayed on the host. The N2048 RUNNING artifact and the unrecovered transport job were not imported.

| artifact | job | status | database run ID |
| --- | --- | --- | --- |
| full-1024-sigma3.json | l4job-0020d65d07154b26a34a94fbf43c9f50 | PASS | `70a26356c92976e448e7faa3e53312829fdeda733dd8b1fadbcf6272abaab247` |
| full-1024-sigma8.json | l4job-0020d65d07154b26a34a94fbf43c9f50 | PASS | `7db4d5c18c8eb8c0d8c0c7075c044391f231321c26a3ac782f8a5cda06c4793c` |
| full-2048-sigma3.json | l4job-0020d65d07154b26a34a94fbf43c9f50 | PASS | `08cdb175f024fe2fcdef93c5f38974e30bb829206165607d83e500036c356952` |
| full-2048-sigma8.json | l4job-0020d65d07154b26a34a94fbf43c9f50 | PASS | `2e19dfaee5e0d801fc44673566e0b7ad5b5c919f826ddfd57854a7d67a2734d4` |
| full-1024-sigma3.json | l4job-2db62713aa16484eb1e717bba9598e8b | FAIL | `224975e1db0b57c9ec63a9d3456e3e61402d7faed7781783dfb984e05d611254` |
| full-1024-sigma3.json | l4job-930479e42193468abd3b5c85e6c92b15 | PASS | `8cf4f08725ce96fba2347af80e5f4743f6bea22c7f6344478ca5c54a0e66c188` |
| full-1024-sigma8.json | l4job-930479e42193468abd3b5c85e6c92b15 | PASS | `a8f66ef3f68b1e9c430427fbcdd505cf8a139beeb8926781a5709b7a6a58ec96` |
| full-1024-sigma3.json | l4job-aebb8a3edf7d48259bbe6eb50923b1bb | PASS | `4f0301db5c8a6f771182acdf7607d8064eed2dd7b87c94bbcfdd1b96e8f817ef` |
| full-1024-sigma8.json | l4job-aebb8a3edf7d48259bbe6eb50923b1bb | PASS | `4ddcb12d382239c2cba1aeb855501268acfeffde97f5f853cd8d9d59519c91a6` |
| full-2048-sigma3.json | l4job-f6160393b0f84b3f8e4d4d37416d607b | FAIL | `0546f58d2ed1179cf94361be9f8447fdea1350fca0e70924d7d48dc873fe64dc` |

Final accounted driver time: 2651.471s of the 7800s stage cap, including the full 900s reservation for the unconfirmed first correctness job. Pool driver timestamps include imports/setup; summary timers start later and therefore differ slightly. No N2048 inverse or further repair run was launched.

Database export copies, insertion/reimport receipts and logs are in ignored `output/torus-hybrid-reuse/database/`. Source/history bundles and evidence manifests are retained locally; the research worktree is kept to preserve needed ignored evidence. Remote branch deletion is allowed only after PR merge.
