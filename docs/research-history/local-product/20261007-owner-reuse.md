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

## 03:01 JST: 短いjobへ分割

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

## 03:20 JST: GPU validation成功

`l4job-f7a31fcb157644a6a15e5e4909d39c7d`:55 passed (54 GPU + declaration)、428.50s。
actual NVIDIA L4 / torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0。
source/result archives、submitted/committed/workerの196 runtime filesのSHA256一致を確認。
6新候補のY/dX/全atom・位置勾配、slices、B1/32/64、N64/N128の20 Graph更新と
Parameter/moments/step、旧forward後のmovement/backward、empty-neighbor範囲を通過。
同名JSONへ検証proofを保存した。
N64測定jobへ進行。GPU test成功は速度やメモリ改善の証拠ではなく、screenを待つ。

## 03:29 JST: 6候補のscreen完了

N64/N128 rho3/8の2 jobsが成功。source/resultと全runtime hash一致、36 full-shape FP64候補比較、各case内の初期Parameters/input hashes一致。各1 executionの21 sample median、単位us。

|size/rho|copy8 baseline|param2|tile32|tile64|tile32 split2|tile64 split2|tile32 param2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|64/3|59.45|52.88|62.97|76.77|68.60|88.74|56.70|39.35|
|64/8|56.51|52.74|61.04|74.38|66.56|84.32|57.03|39.24|
|128/3|70.46|68.02|78.58|91.58|89.53|115.82|76.12|45.74|
|128/8|73.89|69.31|85.97|100.33|98.07|126.24|81.64|46.26|

param2のみが全4条件を改善。allocated peak115712/303104 bytesでcopy8と同じ。partial DQ tensorは存在するが、このprotocolのピークを増やさなかった。tile32/64は全条件で遅く、split2はallocated99328/270336 bytesへ減らすがさらに遅い。採用候補はparam2へ絞る。

N64/rho3のparameter event診断は16.38→10.24us (reduction込み)、compiler218→128 registers、shared24576→20480 bytes、param spill0。N128/rho3は12.29→10.24us、param2のcompiler80 registers/shared40960 bytes/spill2。N64のforwardもcompiler spill2がある。global H allocationは作らないが、spillがある経路を完全なregister/shared常駐と主張しない。

wide output tile64はforward/dXのregisters/shared使用が増え、gridのoutput owner数も1/4になる。N128/rho3 output/dX event13.31/14.34→24.58/25.60us。これらは別graph診断で、原因を一つに断定しない。完全stepの遅延から、この小さい問題ではH再計算削減より仕事分割を優先する。

param VJPのPolar epilogueは各atomのdaへSourceから決まる定数を掛ける線形変換。batch部分和を別々に変換して足しても数学上同じ。FP32丸め差は全4 gradientsと20 updates/momentsのoracleで検証した。

source `facab462` (runtimeは元screenと同じ)でfull rho1.25/3/8/mixedの独立2回を短い4 jobsへsubmitした。copy8/parallel4/param2+dense、repeat2はcase/plan順を逆転する。
- `l4job-fe3049d8391a42cea38725e8e634ef24`:N64 repeat1
- `l4job-5d8f35357380499b8207fa291192f77c`:N128 repeat1
- `l4job-c4efe1c609814295bc519e9aa838b943`:N128 repeat2
- `l4job-c06d9b8c534d47028f74dbd868cb3aa5`:N64 repeat2

4 screen artifactsをDB保存、byte-identical export/idempotent再取込を確認。full/G4結果はまだない。

## 04:25 JST: param2独立2回の完全step測定

4 jobs全て成功。48 full-shape FP64 Y/dX/全atom勾配比較、source/result archivesと196 runtime hashes、各caseの初期Parameter/input bytesのplan間・独立run間一致を確認した。repeat2はcase/plan順を逆転。単位us、2 execution mediansのmedian。

|size/rho|copy8|parallel4|param2|dense|param2 paired短縮率|
|---|---:|---:|---:|---:|---:|
|64-rho1_25|55.56|54.73|50.91|38.98|7.47–9.24%|
|64-rho3|59.32|58.80|52.59|39.05|11.25–11.43%|
|64-rho8|56.63|55.95|52.61|38.91|7.06–7.14%|
|64-rhomixed|62.95|61.83|58.41|38.90|7.13–7.31%|
|128-rho1_25|65.78|70.88|64.26|45.40|2.23–2.38%|
|128-rho3|70.07|75.12|67.49|45.35|3.50–3.88%|
|128-rho8|73.37|78.71|69.23|45.26|5.54–5.74%|
|128-rhomixed|75.75|81.06|71.66|45.31|5.39–5.40%|

allocated peakはN64 115712 / N128 303104 bytes、reserved 6291456 bytesでcopy8と同じ。全atomの幅はproduction optimizerで更新した。denseとの差は残るが、全8条件・両executionでparam2がcopy8を改善。2回の測定から統計的有意性までは主張しない。
screen4＋full16 artifactsをDBへ保存し、byte-identical exportとidempotent再取込を確認。GPU正しさ55 testsは既存proofに記録。G4はこの検証済み候補の短い比較を次に行う。
