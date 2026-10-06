# 2026-10-07: owner preparation の並列化

朝08:00 JSTまで、N64/N128の完全学習step時間と低メモリを改善する研究。
branchは`kernel/owner-prep-parallel`、基点はPR #34の
`5f3d7f540274fa10e7b8d29b10de7c24e63eb214`。
朝までの継続goalと30分間隔のheartbeat `small-linear` を設定した。
新規実験の締切は2026-10-06 23:00 UTC。締切後は回収・GPU停止・報告を終える。

## 第一ラウンドの仮説

前回のordered split4 int32を速度baseline、unsplit orderedをメモリcontrolとする。
rho帯・support開始位置・canonical IDの順番、正規化、全勾配は維持する。
毎forwardの独立snapshotを保存し、可変幅・移動後の支持をその都度構築する。

| suffix | sort warps | owner範囲 | physical payloadコピー | contraction分割 |
| --- | ---: | --- | --- | ---: |
| ordered_split4_i32 (baseline) | 4 | 2 CTA内のownerループ | sortと同じCTA | 4 |
| ordered_prep_parallel | 4 | direction × ownerのCTA | sortと同じCTA | 4 |
| ordered_prep_warp8 | 8 | 2 CTA内のownerループ | sortと同じCTA | 4 |
| ordered_prep_parallel8 | 8 | direction × ownerのCTA | sortと同じCTA | 4 |
| ordered_prep_range | 4 | direction × ownerのCTA | sortと同じCTA | 1 |
| ordered_prep_copy | 4 | direction × ownerのCTA | 範囲構築CTAで分担 | 4 |
| ordered_prep_copy8 | 8 | direction × ownerのCTA | 範囲構築CTAで分担 | 4 |
| ordered_prep_vector | 4 | sort CTA内でowner軸をまとめてreduce | sortと同じCTA | 4 |
| ordered_prep_vector8 | 8 | sort CTA内でowner軸をまとめてreduce | sortと同じCTA | 4 |

ownerは出力または入力の16次元区間を指す。バッチは32行、contractionのbatch tileは16行。
atom blockは32 recordsであり、16次元内に32 atomsが収まるという意味ではない。
並列rangeは現在の支持とbandから重なるphysical区間のmin/maxを計算する。
consumer側の正確な支持maskは維持する。copy候補はcanonical metadataを直接読み、
CTAごとにdisjointなphysical区間だけ書くため、CTA間のコピー結果の読み取りに依存しない。
追加launchの負担を含めて測定する。Hのglobal allocationは作らず、物理DRAM trafficや
L2常駐の改善をこの実装だけから主張しない。

## Protocolと検証状況

`plans-local-prep.json`と`cases/local-prep-{64,128}-rho{1_25,3,8,mixed}.json`。
B32、A204/A819、seed41、FP32 IEEE、TF32 off、21 timing samples。
可変幅のproduction fused AdamW/Polar更新とdenseを同一GPUで比較する。
実際のdecoded初期rhoが全atomで1を超えることをdriver内でassertする。
full-shape FP64 scalar oracleでY、dX、全atom勾配、非零の位置勾配を検証する。
Graph20更新・Parameter/moments/step・slice・支持境界・旧forward snapshotのbackwardを
既存ordered-ownerテストで確認してから測定する。

- `62fc74145f4975900fc8e17feb0823f1b2b9bc12`: 最初の4候補。
  prepare/check全8ケース、ruff、宣言往復成功。
  CPU suite: 860 passed / 525 skipped (26.27s)。skipはGPU検証ではない。
  L4 `l4job-c83dcfc92f2e467e977bee4c786f9592`: 検証とN64/N128 rho3/8 screen、進行中。
- `796dd7434a57a908cf9f02b896538f6941725226`: copy/copy8を追加。
  ruff・宣言往復成功。最初のjobは独立した凍結sourceで実行を継続。
  L4 `l4job-f85284a41832476998f5ff5342f8a0a0`: copy2候補の検証と同条件screen、queued。
- `d6ad1e4f520d0fcbb8839b1a85e1309b4276c7ea`: vector/vector8を追加。
  追加launchなしでowner軸をまとめてreduceする。
  L4 `l4job-94a272bd10584829a87e59948b9836d8`: vector2候補の検証と同条件screen、queued。
  最終8候補の宣言往復、ruff、prepare/check各8ケース成功。
  CPU suite: 860 passed / 557 skipped (23.13s)、wheel/sdist build成功。

## 最初のL4 screen

`l4job-c83dcfc92f2e467e977bee4c786f9592`成功。33 tests passed
(32 GPU + 1 declaration)、24 full-shape FP64 oracle比較成功。
source archive、result archive、submitted/committed/worker runtime source hashを確認した。
簡潔な数値を同名JSONに保存した。各セルは1 executionの21 sample median、単位us。

| size/rho | split4 i32 baseline | parallel4 | parallel8 | warp8のみ | unsplit control | parallel unsplit | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|64/3|59.46|58.42|58.65|60.32|66.59|65.50|38.86|
|64/8|57.07|55.80|56.07|57.83|67.26|65.49|39.08|
|128/3|79.81|75.12|72.45|77.23|94.64|85.93|45.77|
|128/8|82.59|79.08|75.69|80.42|111.22|101.49|45.08|

parallel8はN128で8.4–9.2%改善。N64はparallel4が1.8–2.2%改善。
split4候補のcapture/replay allocated peakはbaselineと同じ115712/303104 bytes。
unsplitは84992/237568 bytesのままで、速度とのtradeoffを維持した。
N128/rho3のlayout event診断は22.53→15.36us (parallel8)、sort registers94→40、
spill0、shared4096 bytes。range kernelは44 registers、spill0、shared16 bytes。
componentイベントは主測定とは別であり、各時間の加算からstep時間を再構成しない。
warp8のみはN128を少し改善するがN64を悪化させる。
まだ独立反復前なので採用判断は保留。rho1.25・mixedを加えた独立2回のL4比較へ進む。

## 再開checkpoint (01:56 JST)

worktree: `/Users/ware10sai/.codex/worktrees/local-product-onchip-h/torchcst`。
raw evidence: ignored `benchmarks/cuda/linear/evidence/owner-prep-20261007/`。
driver、hash確認analyzer、DB保存/byte export/idempotence確認scriptを保存した。
source/result原本はhost-wide poolのjobsディレクトリにも保持する。
pool supervisor: tool session `22376`, one L4, idle-seconds 0。
CPU/buildは完了。最初のscreenが成功、copyがrunning、vectorがqueued。
PR #35 (draft): https://github.com/takumiecd/torchcst/pull/35、base PR #34。
GitHub CPU checks成功(9081f0e0時点)。最初の4 artifactsはDB保存・byte-identical export・idempotent再取込成功。
次はcopy/vectorの結果を回収・hash確認して全screenを同じ表にまとめ、
有望な1–2候補＋baselineでfull8ケースを独立2回測定する。
その後の仮説は32次元output tileでH再計算回数を減らす方式。owner16区間を
結合するときはempty rangeをminから除外し、隣接ownerのmin/maxと正確な支持maskを
維持する必要がある。これは未実装・未測定。新たな実験で位置勾配を失わせない。
他のGPUは停止済み。直接CLIでpool sessionへ介入しない。
再開時はpool statusとjob resultを確認し、同じjobを重複submitしない。

## 02:02 JST checkpoint

copy screen成功:17 tests (16 GPU + declaration)、12 full-size oracle比較成功。
N128 rho3:79.72→70.25us、rho8:82.85→73.72us(copy8)、メモリ同じ303104 bytes。
N64はほぼ同時間。4 raw artifactsをDBに保存、byte export / 再取込を確認した。
vector screen `l4job-94a272bd10584829a87e59948b9836d8` は進行中。
full8ケースの2独立runを、current source `1ea99c9369a73d1d63b766c8b2324b47a5c00dcb`
で凍結しqueued。firstは全ordered-owner121 testsも実行する。
baseline / parallel4 / copy8 / vector8を比較し、secondはcase・plan順を逆転する。
- `l4job-3989854491c041178dfd79a847f148e3`: full repeat1
- `l4job-d9b8fc2194934af0ac1251c4030bb5e6`: full repeat2

次の実装は別branch `kernel/owner-tile-reuse` に分離する。GPU sourceは凍結済みなので
現在の作業treeの変更はこれらのjobへ混入しない。PR #35の完成時にはbranchへ戻って
新しい証拠と説明を追記する。

## 03:00 JST: screen完了と短いjobへの分割

copy/vectorのscreenも成功。各17 tests (16 GPU + declaration)、各12 full-size oracle比較。
最初のscreenと合わせて新候補64 GPU tests、48 full-size比較。12 artifactsすべて
DB保存、byte-identical export、idempotent再取込を確認した。

|size/rho|copy screen baseline|copy4|copy8|vector4|vector8|
|---|---:|---:|---:|---:|---:|
|64/3|59.22|59.15|59.14|70.29|59.06|
|64/8|56.67|56.46|56.66|57.52|56.04|
|128/3|79.72|73.09|70.25|91.17|73.22|
|128/8|82.85|76.57|73.72|83.93|76.39|

単位us、各候補1 execution。vector列は別jobの測定で、copy baselineとの差を
paired改善率と解釈しない。raw screen-all.jsonは同jobのcontrolとの対を保存する。
`ordered_prep_vector`は末尾`_vector`によって既存のconsumer support vectorizationも
有効になる。vector4の結果は準備処理だけの変更ではない。vector8はこの性質を持たない。
N128/rho3 copy8のlayout診断13.31us、sort48 registers/0 spill/4096 shared bytes、
copy/range56 registers/0 spill/16 shared bytes。split4のallocated peakは変わらない。

長いfull repeat1 `l4job-3989854491c041178dfd79a847f148e3` はCLI接続timeoutで
回収不能となった。数値やtest成功を主張しない。poolのowned runtime停止を確認し、
live supervisorがなくなってからrecoverした。full repeat2とreuse/cacheの長いjobは
実行前にcancelしてsourceを保存。kernel失敗とtransport失敗を区別する。

新source `08c87c19` (runtime/tests/toolsは`1ea99c9369a73d1d63b766c8b2324b47a5c00dcb`と同一)
で以下へ分割。one L4 supervisor tool session90589。

- `l4job-7ab7ad83568543f185256a65b683d820`: baseline + parallel/range/warp8検証、41 passed。
- `l4job-4aa79967228b423eb637a08d9a9fb09b`: copy/vector検証、33 passed。
- `l4job-98265f64fd6648cda671d867835ee2ab`: N64 repeat1、測定中。
- `l4job-1de714956be348b3b301d6d7b341bc82`: N128 repeat1、queued。
- `l4job-1a81c7cfeaa744cb8214db5d0e36c1e7`: N128 repeat2、queued。
- `l4job-fadd788094b94c339edb85da33a9c503`: N64 repeat2、queued。

検証2jobのsource/result archive hash、submitted/committed/worker全runtime hash一致を確認。
合計72 GPU tests + declaration 2回。測定jobはfull-shape FP64 oracleを各候補で
通してから同じbaseline/parallel4/copy8/vector8を測る。repeat2はcase/plan順を逆転する。

## 03:05 JST: 最初のN64 full run

`l4job-98265f64fd6648cda671d867835ee2ab`成功。source/result archiveと全runtime hash、
full-shape FP64比較を確認した。4 artifactsをDB保存、byte export/idempotent再取込成功。
隣接JSONを全3screens、最新検証2jobs、このfull runの簡潔な記録へ更新した。

|rho|baseline|parallel4|copy8|vector8|dense|
|---|---:|---:|---:|---:|---:|
|1.25|55.73|54.85|55.50|54.99|39.14|
|3|59.46|58.94|59.23|58.91|39.02|
|8|57.37|56.31|57.08|56.70|41.96|
|mixed|63.32|62.37|62.99|62.43|39.09|

単位us、各1 execution。allocated peakは全候補115712 bytes。rho8のdenseは
他条件より遅いため、独立repeatが揃うまでdense比の結論を保留する。

## 03:06 JST: 最初のN128 full run

`l4job-1de714956be348b3b301d6d7b341bc82`成功、source/result/runtime hash一致。
各ケース内の全候補の初期Parameter/input hashesが一致。新4ケース×4候補のFP64 oracle成功。
各1 execution (us)、allocated peakは全候補303104 bytes。4 artifactsをDB保存、byte export/idempotent再取込を確認。

|rho|baseline|parallel4|copy8|vector8|dense|
|---|---:|---:|---:|---:|---:|
|1.25|75.82|71.50|66.03|69.02|45.71|
|3|79.28|75.29|70.09|73.29|46.14|
|8|83.23|79.17|73.97|77.06|45.85|
|mixed|85.52|81.79|76.18|79.13|45.70|

copy8の狭い条件で約12.9%、mixedで約10.9%の短縮。独立repeat2待ち。
別graphのdiagnosticsでlayoutは22.53–23.55→13.31us。wide/mixedのparameters
診断は16.38usで残り、atom VJPの仕事分割も次の調査対象となる。
イベント時間の合計から完全stepを再構成しない。

## 03:13 JST: L4独立2回の比較完了

4 measurement jobsすべて成功。source/result/submitted/committed/worker runtime hashes一致、各case内の初期Parameters/inputsも全候補・2反復で一致。64 full-shape oracle比較成功。以下は各executionの21 sample medianを独立2回測り、その2 mediansの中央値。単位us。

|size/rho|baseline|parallel4|copy8|vector8|dense|copy8 paired短縮|
|---|---:|---:|---:|---:|---:|---:|
|64/1_25|56.05|54.92|55.40|56.20|39.17|1.16%|
|64/3|59.39|58.93|59.29|58.72|39.11|0.17%|
|64/8|57.44|56.29|56.99|56.56|40.55|0.78%|
|64/mixed|63.21|62.34|63.01|62.37|39.25|0.30%|
|128/1_25|75.50|71.53|66.21|69.03|45.73|12.30%|
|128/3|79.38|75.31|70.14|73.27|47.39|11.65%|
|128/8|83.07|79.20|73.96|77.01|45.84|10.97%|
|128/mixed|85.44|81.83|76.16|79.22|45.78|10.86%|

N128 copy8は各反復で全4条件を改善し、paired短縮中央値10.86–12.30%。N64はparallel4が0.77–2.02%程度と小さい。vector8はN64 rho1.25の2回目を悪化させ、copy8をN128で超えないため現時点の選択肢から外す。

allocated/reserved peaksは全split4候補でN64 115712 / 6291456 bytes、N128 303104 / 6291456 bytes。全raw runで同じreserved peakを確認した。新copy8は追加Hやpartial allocationを作らない。

N128/rho3 denseは46.14/48.64us、N64/rho8 denseは41.96/39.14usと変動がある。dense比とその限界をraw run mediansと併記し、統計的有意差を主張しない。主判断は同じjobのbaselineとcandidateの完全step比較。

全16 full artifactsと12 screen artifacts、合計28 artifactsをDB保存。全件byte-identical exportとidempotent再取込を確認。次は残りfamilyのL4検証/screenを回収し、one-worker poolをdrainしてowned L4停止確認後、copy8/parallel4の短いG4検証・2反復へ進む。
