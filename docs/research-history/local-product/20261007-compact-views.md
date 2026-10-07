# Compact dX snapshots with full forward/parameter metadata

Branch `kernel/compact-ordered-views`, based on merged integration PR #54
(`eff5c277`). Four explicit `_dxview11` variants retain current owner/parameter
partitions and exact cached-order repair. Forward and atom VJP retain all 13
ordered fields. Only the value-only dX direction omits L2-normalizer position
coefficients, using physical mapping `[0,1,2,3,4,5,8,9,10,11,12]`.

The existing parameter VJP reuses forward's ordered physical snapshot, not the
transient canonical preparation buffer. That ownership matters: removing its
normalizer derivative fields breaks positions. The new storage is one flat
`(24,A)` allocation, sliced into forward/parameter `(13,A)` and dX `(11,A)`
views. There is no additional retained canonical buffer or global H. Canonical
atom IDs, immutable per-call snapshots, full-domain L2 normalization, singleton
checks, all source gradients and production AdamW/Polar retain their contract.
The dX helpers' unused second internal result is zero; the actual parameter VJP
always consumes the complete 13-field forward view and normalization derivatives.

Logical copy storage now falls by `8*A` bytes: 1,632 B at A204 and 6,552 B at
A819, half the initial proposal. 24 rather than26 total fields is a7.7% payload
reduction. Allocator peaks and physical DRAM/L2 behavior require measurement.

Host validation uses CPU tests, changed-file Ruff, build and eight fresh
prepare/check comparisons across N64/N128 and initial rho1.25/3/8/mixed (>1).
The new GPU suite checks both directional mappings/support coverage, slices,
batches1/32/64, independent scalar Y/dX/all source gradients and positions,
outstanding backwards, 20 captured optimizer updates, empty neighbors and exact
cached refresh after inversion/end-only changes. GPU validation for this corrected
ownership and selected N128 rho3/mixed timing are pending.

The existing complete-step runner measures current/candidate/dense including
Graph capture/replay allocated/reserved peaks. Reproduce with
`tools.kernel_dev prepare/check`, `plans-local-dxview11.json` and
`local-dxview11-{64,128}-rho*.json`, then `--polar-update fused`.
Ignored driver/analysis/proofs remain in
`benchmarks/cuda/linear/evidence/compact-views-20261007/`. Public dispatch is unchanged.

## Initial GPU compile failure

The first frozen source `f6a53b66` failed GPU compilation: a loop-local
`canonical: tl.constexpr` was assigned more than once during static unrolling.
The run (`l4job-cb0bf232b99a4bb4906bb44bf36016c9`) returned exit1/no timeout,
67 compile-dependent failures and4 passes, before any measurement. Source/result
archive hashes and the failure logs are preserved under ignored evidence. It is
excluded from correctness and speed claims. Commit `05509955` removes the
reassignment restriction without changing the physical mapping; the repaired
validation is `l4job-ab77a2e89f4f4a9d893918c008c88e03` on the same owned L4.
The earlier queued job33b4 was cancelled before execution to correct its source
label; it consumed no GPU experiment and is also excluded.

## Both-directions compact design rejected by gradient checks

Repaired-copy source `05509955` compiled and passed43 checks (including existing
13-field routes and new support/snapshot checks), but28 new gradient/update
checks failed. The parameter tensor comparison includes center gradients;
Y/dX alone does not certify the design. Inspection then confirmed the autograd
context saves forward's ordered view as parameter metadata. Both-directions
compaction had removed those normalization derivative fields. No tolerance was
relaxed and no performance measurement ran.

Job `l4job-ab77a2e89f4f4a9d893918c008c88e03`, complete failure logs and hashes are
preserved/excluded. The design was changed to full forward/parameter13 plus dX11,
so the existing VJP can reuse its complete physical view without a new retained
canonical allocation. The original16*A payload estimate was invalid for that
ownership and is not a measured low-memory result.
