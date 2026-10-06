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
