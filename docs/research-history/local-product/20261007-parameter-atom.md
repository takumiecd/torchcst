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
