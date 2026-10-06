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
