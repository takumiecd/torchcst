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
