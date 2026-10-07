# Compact ordered value snapshots

Branch `kernel/compact-ordered-views`, based on merged integration PR #54
(`eff5c277`). Four explicit `_view11` variants retain the existing owner/parameter
partitions and exact cached-order repair choices. Only the ordered forward/dX
copies omit canonical fields 6/7, the two L2-normalizer center derivatives.
Canonical `packed` retains all 13 fields for parameter VJP, including positions.

Physical mapping is `[0,1,2,3,4,5,8,9,10,11,12]`. Compact support bounds shift by
two fields; owner ranges read canonical metadata during parallel copying. Each
call still owns immutable physical snapshots and canonical IDs. No H allocation,
normalization, full-domain singleton test, source-width VJP or production
AdamW/Polar update rule changes. The compact factor helpers are value-only:
their unused second internal result is zero, while actual coordinate/source
VJPs always use full canonical metadata through the existing parameter path.

Logical copy storage falls by `16*A` bytes across both directions: 3,264 B at
A204 and 13,104 B at A819. This is a payload calculation; allocator peaks and
physical DRAM/L2 behavior require separate measurement. No speed claim yet.

Host: 894 CPU tests pass, 1,054 GPU/DB skips; changed-file Ruff and whitespace
checks pass. Eight fresh prepare/check comparisons cover both N64/N128 and
initial decoded rho1.25/3/8/mixed (>1). The new GPU suite checks compact field
mapping/support coverage, slices/B1/32/64, independent scalar gradients,
outstanding backwards, 20 captured optimizer updates, empty neighbors and
cached-order inversion/end-only refresh with immutable snapshots.

GPU correctness and timing pending. The first selected measurement targets
N128 rho3/mixed, with current repair4/8 counter-free controls, matched compact
variants, copy8 baseline and dense. The existing complete-step runner includes
Graph capture/replay allocated/reserved peaks. Further work follows retrieved
evidence, with negative results retained.

Reproduce declarations with `tools.kernel_dev prepare/check`, catalog
`benchmarks/cuda/linear/plans-local-view11.json`, and Cases
`benchmarks/cuda/linear/cases/local-view11-{64,128}-rho*.json`.
Actual measurement uses the existing Linear runner with `--polar-update fused`.
Ignored driver/analysis and source/result proof live in
`benchmarks/cuda/linear/evidence/compact-views-20261007/`; raw evidence is not
committed. Public dispatch remains unchanged.
