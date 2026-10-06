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
