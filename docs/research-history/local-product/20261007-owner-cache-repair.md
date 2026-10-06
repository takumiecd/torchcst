# 2026-10-07: exact bounded neighboring-order repair

Branch `kernel/order-local-repair`, based on PR #41 `a029acfe`.

PR40のN128 rho3では52 refresh中49/50回に実際の逆転があり、検査だけでは全sortを
省けなかった。今回はcached順のcurrent full keysに隣接compare/exchangeを
4/8 rounds (各roundはeven/oddの2 layers)適用する。逆転が残れば同じfullキーの
通常sortへfallbackする。近傍修復が成功した場合だけfull sortを省く。

## 正確さ

key `k(a) = (bucket(a)*(max(K,N)+1)+support_lo(a))*C+a` はcanonical tie-break
付きで一意。各layerのpairはdisjoint、両側のmin/max交換はキーのpermutationを保存。
odd layerの先頭・末尾はself partnerで保存。paddingをsentinelへ戻してから交換し、
cached IDsのpaddingに置いたID0が有効キーを複製しないようにする。
全layer後に隣接逆転を検査し、なければfresh full sortと同じ唯一の順序。
残ればfull sortするため、movementが大きい/geometryが混在する場合も近似しない。

canonical連続metadata readとキーgatherはPR41と同じ。bucket histogram/offset、
normalizer・支持終端・owner envelope・係数、forwardごとの独立snapshotは毎回更新。
位置/dX/全atom gradients・singleton・production可変幅optimizerを維持する。
追加stateはrepair/full-sort counters32 bytesのみ。Hをglobal保存しない。
物理L2/DRAM改善やGPUピーク改善の証拠はまだない。

## Host checkpoint 04:52 JST

881 passed /679 skipped (22.92s)、changed-file ruff、5 CPU declaration/boundary
checks、8 prepare/check、isolated wheel/sdist build成功。GPU正しさ・速度は未検証。
22 GPU testsは既存FP64 gradients/slices/B1/32/64、20captured optimizer updates/
moments/steps、old snapshots、両/片方向のinversionsに加え、隣接交換だけで修復する
場合とlast IDの66位置movementで必ずfull-sort fallbackする場合をfresh stateless
layoutとexact比較する。order_updates counterは修復とfull-sortの合計、追加counterは
bounded_repairs/full_sortsを区別する。元routesのcounter意味は変えない。

性能はcopy8・PR41 gather copy8のcontrols、repair4/8とdenseを同時比較。
N64/N128 B32 A204/A819、seed41、初期rho>1の1.25/3/8/mixed、FP32 IEEE/TF32off、
production fused AdamW/Polar、21 samples。L4 frozen jobsで検証する。
raw scripts/resultsはignored evidence/owner-cache-repair-20261007。

G4 selected2候補の独立2runは完了し、supervisor45682 terminal exit0、全slots stopped、
lifecycle Session terminated/server No active sessionsを確認。新L4 supervisorで
先にPR41の既存queued3 jobsを実行し、この新方式を続ける。

source `ae2e5faeaa2272f0cfd47c64c6df62cddc1cfa45`、draft PR #42。check `l4job-cfc1bdee05504cf885de1a7cbd3ae271`、N64 `l4job-f13a06ef293c4578a3ae90b6231a6c5d`、N128 `l4job-d0b94b21542f4aa49eac340e1d427ef5` queued。L4 supervisor tool97800でPR41のcheck/N64/N128→この3 jobsの順に実行。source snapshotsは独立で、その後のbranch editsを含めない。

## 05:22 JST: full sort削減をGPUで確認

check27 passed (22 GPU +5 CPU)、27.99s。actual L4 /torch2.11+cu130/CUDA13/Triton3.6、source/resultと197 runtime hashes一致。N64/N128 jobs成功、32 full-shape FP64比較とcase内初期Parameters/input bytes一致。各1 execution /21 samples、us。

|size/rho|copy8|gather copy8|repair4|repair8|dense|
|---|---:|---:|---:|---:|---:|
|64-rho1_25|55.23|54.46|54.51|54.42|39.12|
|64-rho3|58.98|62.48|59.00|59.65|39.34|
|64-rho8|56.52|55.88|55.94|55.91|39.27|
|64-rhomixed|62.76|62.76|62.59|62.25|39.07|
|128-rho1_25|66.01|62.54|62.72|62.55|45.22|
|128-rho3|70.22|71.14|72.22|68.46|45.80|
|128-rho8|73.79|70.52|70.67|70.60|45.89|
|128-rhomixed|75.70|76.70|73.62|74.20|45.61|

N128/rho3:52 refresh中gatherのforward49/dX50 full sortsをrepair8で1/1へ減らし、48/49回はbounded repair成功。layout event14.34→12.29us、完全step70.22(base)/71.14(gather)→68.46(repair8)。repair4は26/34 full sortsが残り、repairとfallbackの両方を払って72.22usと遅い。
N128/mixed:repair4はforward37 repairs＋6 sorts、dX37＋7でlayout14.34→11.26us、complete75.70→73.62us。repair8はfull sorts0/0だが16layersの費用で74.20us、少ないsortが常に最速ではない。rho8 repair4はほぼ全fallback、repair8は4/3 sortsへ減るがtime70.60usでgather70.52usと近い。
N64/rho3:repair4/8は26/25 changesを全て修復できたが、base58.98とrepair4 59.00usはほぼ同じ。gather control62.48usは先のscreen59.66usより遅い変動を含むので、controlだけに対する差を一般化しない。
allocated117248/307200、reserved6291456でgatherと同じ、no-cache115712/303104より高い。Hのglobal allocationは増やさず、layout compiler spill0。ただし全kernelの完全on-chip常駐や物理DRAM削減は主張しない。
近傍修復は探索削減の証拠がありN128 middle/mixedで有望。次にparameter split2と組み合わせ、全rho独立2回で測る。8 artifacts全てDB byte-identical export/idempotent再取込成功。
