# 2026-10-07: parameter VJPのatom tile

Branch `kernel/parameter-atom-tiling`、PR #36 checkpoint `683061d0`を基点とする。
copy8のlayout改善後、N128 wide/mixedでparameter診断が16.38us残っている。
これは別graphのevent診断であり完全step時間とは加算しない。

## 仮説と仕事の軸

N128/A819のparameter VJPは32 atom records/CTAで26 blocks、N64/A204は7 blocks。
parameterだけ16 records/CTAにすると52/13 blocksになる。各atomの4勾配はそのCTAが
完全に計算するので、batch分割VJPとは異なり追加の部分和tensorやreductionは不要。
contractionの32 atom records、16 batch行、16出力次元、owner4分割は維持する。

`Recipe.parameter_atom_block`のdefaultは既存の`atom_block`。新routeだけ16にする。
既存と同じparameter kernelへgridとBAを渡し、4 warps/既定warpsも比較する。
各atomの全batch/支持、正規化、位置VJP、canonical ID scatter、widthのproduction
AdamW/Polar更新は維持する。FP32 reductionの丸め差は独立FP64 oracleと20更新で調べる。
Hのglobal tensorや新しい永続tensorは追加しない。メモリ/速度の効果は未測定。

|suffix|parameter atom records|parameter warps|parameter batch splits|
|---|---:|---:|---:|
|ordered_reuse_paramatom16|16|既定(N64:4/N128:8)|1|
|ordered_reuse_paramatom16_param4|16|4|1|
|ordered_reuse_paramatom32_param4|32|4|1|

controlsはcopy8 baselineとPR #36のparam2 (batch2分割VJP)。
N64/N128 B32 A204/A819、seed41、FP32 IEEE、TF32 off、21samples/execution、
初期decoded rho>1、rho1.25/3/8/mixedの同一fixture。まずrho3/8をscreenする。

## Host検証

変更対象のruff、宣言往復、8 cases prepare/check、wheel/sdist build成功。
CPU suite:862 passed / 638 skipped (23.14s)。skipはGPU成功を意味しない。
新27 GPU tests + declarationを用意。実際のY/dX/全atom勾配、slice、B1/32/64、
N64/N128の20 Graph更新・moments/step、旧forward snapshotのbackward、
singleton/middle/wideとempty-neighborを独立oracleで比較する。
GPU validation/screenは未実施。raw driver/analyzer/DB scriptsはignored
`benchmarks/cuda/linear/evidence/parameter-atom-20261007/`に保存。

広いruff checkでは変更していない`src/torchcst/__init__.py`のI001/RUF022が残る。
HEADと作業treeのbyte一致、HEAD入力にも同じ2件が出ることを確認した。
この研究branchでは別scopeのformat変更を加えない。

## 03:12 JST: L4 queued

source `7cbb318b20e788b1fc8d6a54ec5db57bc3880e84`から凍結してone-worker queueへsubmit。

- `l4job-701b166db3da4900a8f488b3cdbf031b`: 28 tests (27 GPU + declaration)。
- `l4job-5bf52062e16c48e99ba71c1695e15076`: N64 rho3/8、3候補+2controls+dense。
- `l4job-656ac889dc834f60885bb206cf3440ae`: N128 rho3/8、同じ比較。

full-shape FP64 Y/dX/全atom oracleを各候補で通してからcomplete stepを測る。
GPU結果は未回収。新候補の位置勾配・20更新・peaksが揃うまで採用を決めない。

## 03:50 JST: fixture修正と最初のscreen

旧GPU job `l4job-701b166db3da4900a8f488b3cdbf031b`は25 passed (24 GPU +
declaration)、3 failed。失敗はempty-neighbor testがreuse caseとparamtile catalogを
組み合わせた参照ミス。kernel assertionではないが全件成功とも扱わない。
`ce117f28c86b3167ee5c4ce48d42509643740ff6`で専用caseをmodule collection時に読み、
CPU collectionでもcatalog一致を検査するよう修正。runtime sourceは変更しない。
修正後28 testsを`l4job-4567ca9dcf7c4d8294fe0634c8826b3c`へ再提出した。

2 screen jobsは成功、source/result/runtime hashesと20 full-shape FP64比較を確認。
初期Parameters/inputsも各case内の全候補で一致。各1 executionのmedian(us):

|size/rho|copy8|param2 control|param atoms16 default warps|atoms16 warps4|atoms32 warps4|dense|
|---|---:|---:|---:|---:|---:|---:|
|64/3|59.52|52.66|53.06|53.03|59.50|39.26|
|64/8|56.41|56.41|51.66|51.66|57.07|39.16|
|128/3|70.18|67.86|68.64|69.00|72.02|45.58|
|128/8|74.10|69.54|71.90|70.78|78.40|46.11|

N64 atom16はrho3/8で約11%/8%改善、allocated115712 bytesは同じ。N128は
param2が速いがatom16も改善する。atoms32 warps4はN128を悪化させる。
N64/rho8のparam2 controlは前screen52.74usから56.41usへ変動。初期p hash、
compiler register/spill/sharedとparameter event診断10.24usは一致した。
完全stepの独立反復が必要で、component時間から結果を置き換えない。
4 screen artifactsをDB保存、byte-identical export/idempotent再取込を確認。GPU全28-test再検証は進行中。速度候補の採用はその後。

## atom16のfull反復を追加

source `ce117f28` (runtimeは初回screenと同じ)からrho1.25/3/8/mixedを独立2回。
copy8 baseline/param2 control/atom16 warps4+dense、2回目はcase/plan順を逆転。
GPU28-test再検証とすべてのfull-shape oracle結果を揃えてから採用判断する。


- `l4job-affdcad5f887440499f30ae86c0cc58f`:N64 repeat1
- `l4job-f67e722e317b4792a98ec9b9d06f8674`:N128 repeat1
- `l4job-139567da57814699921195df68a7263e`:N128 repeat2
- `l4job-b17c0d69b92e46e39c3c1e5ff9c34cc5`:N64 repeat2

## 04:31 JST: 全GPU recheck成功

`l4job-4567ca9dcf7c4d8294fe0634c8826b3c`:28 passed (27 GPU + declaration)、77.75s。source `ce117f28`、actual NVIDIA L4 / torch2.11.0+cu130 / CUDA13 / Triton3.6。source/result archivesと196 runtime filesのsubmitted/committed/worker hash一致。fixture catalog修正後に3 empty-neighbor checksも成功。独立2回の全rho完全step測定は進行中。G4ではcopy8・検証済みparam2・atom16param4のrho3/mixed 4条件だけを2回測る。

G4 frozen source `8247f9e3`。check `colabjob-3e262ce00e2c478c906e26b35931c04b` queued。最初のmeasurement2 jobsはatom16 alias誤記をsubmit後検出し実行前cancel、数値なし。driverはselected aliasesの集合検査を追加し、正しいcatalog IDsで再submitした。L4 supervisorはG4を起動せず、L4 batch停止確認後に明示G4 supervisorで短いbatchを実行する。

G4正式measurement IDs: `colabjob-e698cde57c6d401f848e06ba579cd64f` repeat1、`colabjob-12c93a616fcb4dcfbd0c2a1c95bc24ae` repeat2。actual RTX PRO 6000 Blackwell / CC12.0をdriverでもassertする。

## 04:34 JST: 全rho独立2回完了

source `ce117f28`、4 full jobs成功。48 full-shape FP64候補比較、source/result/196 runtime hashes、caseごとの初期Parameters/input bytesが全plan・両executionで一致。21 samples/execution、repeat2順序逆転。下表は2 execution mediansのmedian、us。

|size/rho|copy8|param2|atom16/warps4|dense|
|---|---:|---:|---:|---:|
|64-rho1_25|55.08|50.72|50.48|38.95|
|64-rho3|59.38|52.56|53.02|39.15|
|64-rho8|56.60|52.54|51.29|38.97|
|64-rhomixed|63.01|58.30|57.69|38.81|
|128-rho1_25|65.84|64.39|64.84|45.42|
|128-rho3|69.89|67.49|68.84|45.44|
|128-rho8|73.65|69.48|70.48|45.34|
|128-rhomixed|76.23|71.65|72.26|45.41|

atom16はcopy8を全8条件・両executionで改善。N64/rho8とmixedではparam2をさらに短縮、rho3ではparam2が速い。N128は全4条件でparam2がatom16より速い。単一構成を全条件の勝者とせず、サイズ・rhoで候補を選ぶ。
allocated peakは115712/303104 bytes、reserved6291456 bytesでcopy8と同じ。atom16はbatch partial DQを追加せず、param2はpartial/reduction込みで同ピーク。物理DRAM trafficやcache常駐は未計測。独立2回で統計的有意性は主張しない。DB16 full artifactsのbyte-identical exportとidempotent再取込が成功。screen4と合わせ20 artifactsを保存。G4短比較はL4 batch drain後に実行する。

parameter compiler診断: N64 atom16/warps4は127 registers/shared16384/spill0、param2は128/shared20480/spill0、copy8は218/shared24576/spill0。N128 atom16は96/shared24576/spill0、param2は80/shared40960/spill2。N128はspillのないatom16よりparam2が完全stepで速いので、spill個数だけで採否を決めない。別graphのevent診断は原因の参考に留め、完全stepの数値で評価する。

## 04:56 JST: G4独立2回の短比較完了

actual NVIDIA RTX PRO 6000 Blackwell Server Edition /CC12.0 /188 SM、torch2.11.0+cu130 /CUDA13.0 /Triton3.6、TF32off。source `8247f9e3`。atomtile27 GPU+declaration tests28 passed (47.66s)、source/resultと196 runtime hashes一致。2 measurement jobs成功、24 full-shape FP64候補比較・hashes、G4内plan/独立run間の初期Parameter/input bytes一致。GPU間の初期Parameter bytesが同じとは主張しない。2 execution mediansのmedian、us。

|size/rho|copy8|param2|atom16/warps4|dense|
|---|---:|---:|---:|---:|
|64-rho3|55.29|50.55|50.82|33.28|
|64-rhomixed|58.02|54.94|54.40|33.34|
|128-rho3|63.20|61.19|62.30|42.80|
|128-rhomixed|69.54|65.65|66.22|42.72|

両候補とも全4条件・両executionでcopy8を改善。param2はN128両条件とN64/rho3で速く、N64/mixedはatom16が速い。allocated peak115712/303104、reserved6291456 bytesでcopy8と同じ。denseとの差は残る。
G4の28 testsはatomtile3候補のreference optimizer更新を含む。param2はG4 full FP64とbenchmarkのlive width更新を通過したが、reference20 updates/moments/stepsはこのsuiteに含まれない。測定と同じruntime sourceからparam2だけの9 GPU testsを短い後続G4 probeへsubmitした。L4でのparam2全reference checksは既に成功。
G4 supervisor45682 terminal exit0、slots stopped、Session terminated /server No active sessionsを確認後、新L4 supervisor97800でgather/repair検証へ戻った。G4 measurement8 artifactsのbyte-identical export/idempotent再取込成功、family合計28 artifacts保存。

後続G4 reference probe `colabjob-5527c4b067834aca947755c81bbb1418`、source `74d1baf7` (runtimeは測定source8247と同じ)、selectorでparam2のみ9 testsをhost collection確認。L4 batchが終わって停止を確認後、G4でこの短い検証を行う。

G4 param2の追加reference probe `colabjob-5527c4b067834aca947755c81bbb1418` は9 GPU tests passed (40.67s)。N64/N128 20 captured reference optimizer updates/moments/steps、slices、B1/32/64、old backward、empty-neighborを検証。source74d1baf7とsubmitted/workerの196runtime files・archive hashesを照合、実GPU Blackwellをproof JSONで確認。supervisor65921 terminal0、slot stopped、Session terminated/server No active sessions確認済み。
