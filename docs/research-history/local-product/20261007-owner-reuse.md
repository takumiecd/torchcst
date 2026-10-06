# 2026-10-07: wider owner tiles / parameter VJP partitions

Branch `kernel/owner-tile-reuse`, based on PR #35 checkpoint `08c87c19`。
PR #35の準備最適化は別branchで継続検証している。
この変更はnormalized local-productの演算・勾配を維持し、Hをglobal bufferへ保存しない。
朝08:00 JSTの新規実験締切は引き続き適用する。

## 仮説と実装

N128のcopy8準備は最初のscreenで13.31usまで短縮した。
残るoutput / dXとパラメータ勾配を狙う。まだ今回の速度改善は未測定。

| suffix | 出力次元/block | atom chunk分割 | parameter batch分割 |
| --- | ---: | ---: | ---: |
| ordered_reuse_tile32 | 32 | 4 | 1 |
| ordered_reuse_tile64 | 64 | 4 | 1 |
| ordered_reuse_tile32_split2 | 32 | 2 | 1 |
| ordered_reuse_tile64_split2 | 64 | 2 | 1 |
| ordered_reuse_param2 | 16 | 4 | 2 |
| ordered_reuse_tile32_param2 | 32 | 4 | 2 |

各候補はcopy8の位置順・rho帯・canonical ID・独立snapshotを使う。
consumerのbatch tileは16行、atom blockは32 records。出力tile32/64は、
H計算を32/64出力次元へ使い回すためのサイズであり、入力次元を選抜する数ではない。
出力tileを広げるとconsumer CTA数は減る。atom分割2/4を合わせて並列度と重複H計算を比較する。

一般atom範囲は16次元ownerの2/4区間を結合する。empty [0,0]をminから除外して
同じrho帯以外のrecordsを取り込まず、consumerの正確な支持maskを維持する。
sliceの最後のownerはmaskし、非単調な支持終端にもmin/max envelopeで対応する。

parameter分割は16 batch行ずつのchunkを2 CTAへ交互に分担させ、各CTAが
振幅・入力中心・出力中心の寄与と既存Polar pullbackを計算する。
canonical ID位置の独立partial tensorへ書き、最後に加算する。
どちらの分割もoptimizerは既存の1回のproduction AdamW/Polar更新を使う。
余分なglobal partial allocation / reductionの費用を完全stepで測る。
shared/register spillや物理trafficは実測なしに改善と主張しない。

## Protocol / checkpoint (02:10 JST)

source `140562ce8d36a81d7d98a6038cbeb1abfa56f179`。
`plans-local-reuse.json`のcontrolsはcopy8、parallel4、元のsplit4/i32。
N64/N128、B32、A204/A819、seed41、FP32 IEEE、初期rho>1の1.25/3/8/mixed。
21 samples、production可変幅fused AdamW/Polar、dense同時比較。

ruff、宣言往復、8ケースprepare/check成功。
CPU:861 passed / 611 skipped (22.02s)、wheel/sdist build成功。
PR #36 draft: https://github.com/takumiecd/torchcst/pull/36。
GPU testsは既存slice・repeated backward・20 Graph updates・旧snapshot backwardを再利用し、
empty隣接ownerとsingleton/middle/wideの混在を独立FP64 oracleで追加検証する。
性能fixtureとsingletonテストの初期幅を区別する。
L4 `l4job-cf4ed3f8c10844d78a1bcee71f5f7401`はqueued、
new reuse tests55 + parent ordered-owner tests121の後、rho3/8の4ケースを測る。
完了を推測せずresultとhashを確認してから有望案を絞る。

raw evidenceはignored `benchmarks/cuda/linear/evidence/owner-reuse-20261007/`。
driver/analyzer/DB scriptを保存した。pool supervisor tool session `22376` は
one L4でPR #35のfull repeat1→repeat2→このscreenを順次実行する。
PR #35のfull sourceは`1ea99c9369a73d1d63b766c8b2324b47a5c00dcb`で凍結しており、
現在のkernel変更は混入しない。PR #35の結果回収時はそのbranchへ戻ってノートを更新する。

PR #35のvector4候補は`vector_support=True`も含むことを現在のrecipe検査で確認した。
範囲構築単独の比較として扱わず、vector8 / copy8はこのflagがFalseである。
既存aliasの測定契約は変更していない。後続報告にはこの交絡を明記する。

## 03:05 JST: 短いjobへ分割

長い親jobが接続timeoutで回収不能となったため、未実行の
`l4job-cf4ed3f8c10844d78a1bcee71f5f7401`をcancelした。数値はない。
source `aad657e4d029a61bfea437ac3b4c88d0cb42c5f7`
(runtimeは`140562ce8d36a81d7d98a6038cbeb1abfa56f179`と同一)から、
L4 one-worker queueへ以下をsubmitした。

- `l4job-f7a31fcb157644a6a15e5e4909d39c7d`: 新候補55 testsのみ。
- `l4job-282e9895051b41ddb37cc5e468f6f0e0`: N64 rho3/8、6新候補と3controls。
- `l4job-a0d84b85e3b649139b497bb63dffce9c`: N128 rho3/8、同じ比較。

GPU検証・screenは未完了。各測定はfull-shape FP64 Y/dX/全atom勾配oracleを
通してから行う。初期rho>1、production width更新、21 samples、denseも維持する。
