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
