# 2026-10-07: canonical metadata reads for exact order validation

Branch `kernel/order-validation-gather`, based on PR #40 GPU-validated checkpoint `28ffa447`.

PR40はcached IDsを使って6つのmetadata fieldsを間接loadし、current full keysを
cached順で計算する。今回はcanonical ID順に同じfieldsを連続loadして全キーを作り、
キーだけを `tl.gather(key, cached_IDs)` で並べて逆転を検査する。
metadataの読み出し順とキーの再配置費用を比較する。物理traffic/L2/DRAMは未測定。

unique canonical tie-break付きfullキーの隣接逆転がなければ元のcached permutationは
fresh full sortと等しい。逆転があればcanonical full keysをsortしID cacheを更新。
現在のbucket histogram/offset・normalizer・support ends・owner envelope・係数と
per-forward snapshotsは毎回更新。位置/dX/全atom勾配、singleton、production可変幅
optimizerを維持。新state tensorもglobal Hも追加しない。

新routesはgather copy8/range8、比較controlsはcopy8とPR40 validated copy8。
output16次元owner、batch16行、atom chunk32、owner split4。ID-only cacheはint16/32
lossless、key/offset cached buffersは空。snapshot Ordersはint32を維持する。

Host: 5 declaration/boundary tests passed /20 GPU skipped、8 prepare/check成功、
changed-file ruff成功。全CPU suite876 passed /657 skipped (22.65s)、isolated wheel/sdist build成功。GPU正しさ・性能は未検証。
20 GPU testsはPR40と同じ数学・slices・20captured live updates/moments/steps・
old backward snapshots・direction-specific inversionsを独立oracleで確認する。
性能はN64/N128 B32 A204/A819 seed41、初期rho>1の1.25/3/8/mixed、
FP32 IEEE /TF32off、production fused AdamW/Polar、21 samples、dense同時比較。
現在のL4 batch drain後にG4 selected comparisonを先に実施し、次のL4 batchで検証する。
raw scripts/resultsはignored evidence/owner-cache-gather-20261007へ保存する。

## 05:20 JST: GPU検証と全rho screen回収

source `a029acfe`、check `l4job-36b509d9f7a04e068fef04350c2b6ce9`:25 passed (20 GPU +5 CPU)、98.44s。actual NVIDIA L4 /torch2.11+cu130/CUDA13/Triton3.6。source/resultと197 runtime hashes一致。N64 `l4job-53b209fddb6f4c0c89aaf8f55f5caa18` /N128 `l4job-75a98df9b8d040879c13fcfc364f971b` 成功、32 full-shape FP64比較とcase内初期Parameter/input bytes一致。各1 execution /21 samples、us。

|size/rho|copy8|validated copy8|gather copy8|gather range8|dense|
|---|---:|---:|---:|---:|---:|
|64-rho1_25|54.93|54.49|54.55|54.17|39.19|
|64-rho3|59.26|59.68|59.66|59.53|39.05|
|64-rho8|56.51|56.12|56.14|55.42|39.23|
|64-rhomixed|62.77|63.00|62.29|62.22|39.36|
|128-rho1_25|66.16|62.68|62.75|64.63|45.78|
|128-rho3|69.78|72.77|71.08|73.44|45.76|
|128-rho8|73.45|73.77|70.26|72.60|45.58|
|128-rhomixed|75.59|78.47|77.18|79.27|45.81|

allocated peaks117248/307200、reserved6291456 bytesでvalidatedと同じ、base115712/303104より高い。N128 middle/mixedはvalidatedより短縮するがcopy8より遅い。N128 rho8のvalidated control73.77usは先のscreen69.90usより遅く、単一executionの変動として保存し他の値で置き換えない。全条件の改善を主張しない。
metadataのcanonical連続読み出しという構造は維持できたが、それだけでソート問題を解消しなかった。続いて近傍修復で実際のfull sortを減らす。8 artifacts全てDB byte-identical export/idempotent再取込成功。
