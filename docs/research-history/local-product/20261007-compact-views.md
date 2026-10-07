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
A819, half the initial proposal. 24 rather than 26 total fields is a 7.7% payload
reduction. Allocator peaks and physical DRAM/L2 behavior require measurement.

Host validation passed: CPU 894 passed / 1,054 skipped, changed-file Ruff,
wheel/sdist build and eight fresh
prepare/check comparisons across N64/N128 and initial rho1.25/3/8/mixed (>1).
The new GPU suite checks both directional mappings/support coverage, slices,
batches1/32/64, independent scalar Y/dX/all source gradients and positions,
outstanding backwards, 20 captured optimizer updates, empty neighbors and exact
cached refresh after inversion/end-only changes. GPU validation for this corrected
ownership passed 71 checks (69 CUDA plus 2 declarations) in 177.28s on NVIDIA L4. Frozen source
`0f79fa60ad7fa79617b2547f5f520a8787c58a19`, validation/measurement job
`l4job-0b03e028784f494dbe4a6fc36372522f`; source/result archives and all
197 committed/submitted/worker runtime files match. First N128 rho3/mixed
measurements pass the full-shape FP64 oracle including nonzero center gradients.
Independent reverse-order N128 repeats and selected N64 uncached parameter-atom16 comparisons are complete. Their current/candidate initial parameter bytes and all input/target bytes agree; decoded initial rho is >1 and widths change under production updates. All 32 full-shape plan comparisons pass Y/dX/all atom-gradient FP64 checks, including nonzero center derivatives. Maximum absolute errors are 1.34e-6 / 1.46e-6 / 4.47e-6 for Y/dX/P.

## Paired complete-step results

B32, N64/A204 and N128/A819, rho3/mixed, FP32 IEEE/TF32 off. Forward, backward and fused production AdamW/Polar are included. Two independent executions per case, 21 samples each; the second reverses case and plan order. Values below are medians of execution medians. Positive time reduction means faster. N64 uses the matched uncached parameter-atom16 control; this is not a new comparison against every cached64 alternative.

|N / rho / matched route|Current µs|dX11 µs|dense µs|Capture/replay allocated B|Paired time reduction|
|---|---:|---:|---:|---:|---:|
|128 / 3 / repair4|68.98|68.71|45.44|306,688 → 300,032|0.38%|
|128 / 3 / repair8|65.53|67.26|45.44|306,688 → 300,032|-2.63%|
|128 / mixed / repair4|68.78|68.85|45.07|306,688 → 300,032|-0.11%|
|128 / mixed / repair8|69.07|69.21|45.07|306,688 → 300,032|-0.21%|
|64 / 3 / param16|51.10|50.72|38.92|115,712 → 114,176|0.73%|
|64 / mixed / param16|56.74|56.20|39.05|115,712 → 114,176|0.96%|

Allocated savings repeat exactly: 1,536 B at N64 and 6,656 B at N128 (allocator rounding differs from the logical payload). CST reserved peak remains 6,291,456 B; dense reserved peak 48,234,496 B. Dense allocated is 34,179,584 B (N64) / 34,408,960 B (N128), including workspace rather than just model weights. Dense uses independent nn.Linear initialization; shape/input/target/dtype and timing boundaries agree, but initial matrix and update trajectory differ.

N64 gains are small but have matching direction in both executions: rho3 0.70%/0.76%, mixed 1.15%/0.77%. N128 repair4/rho3 improves 0.59%/0.17%, but repair4/mixed changes 0.15%/-0.37%. Repair8/rho3 regresses 0.61%/4.65%; repair8/mixed is -0.45%/+0.02%. Therefore retain current full13 dX for N128 speed priority; keep dX11 as an explicit low-memory research option. Public dispatch is unchanged. No G4 sweep was warranted by these L4 results.

All four executions use NVIDIA L4 UUID GPU-8bfc06cf-e0fc-f11d-01b7-18018e0e713c, driver580.82.07, Torch2.11.0+cu130, CUDA13.0 and Triton3.6.0. Source/result hashes, per-execution timings and paired deltas are in [the concise results](20261007-dxview11-results.json). Detailed samples/compilers/raw snapshots remain ignored. No physical DRAM/L2 claim; instrumented phase timestamps are diagnostic only and are not summed into complete-step timing.

## Next implementation direction

The measured field reduction changes full-step time by less than 1% on 64 and does not consistently improve 128. Next prioritize eliminating preparation/layout passes or reducing parameter-VJP work while retaining exact geometry and forward/parameter snapshot ownership. This is a design direction, not yet a measured or implemented speedup.

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


## Persistence and cleanup

All eight complete-step artifacts were imported to the benchmark DB, exported with identical bytes and reimported idempotently. Source/result archives for successful and failed runs, raw logs, original rejected commits and branch checkpoints remain available; no worktree cleanup occurred. Supervisor42474 exited0; all pool slots are stopped, and the lifecycle log confirms Session terminated / No active sessions found on server. No pending experiment remains in this campaign.

## Integration disposition

PR55 incorporates current main and PR46's parameter-batch-loop history. The
combined-source gate and case-scoped speed/memory policies are recorded in
[the integration report](20261007-pr-integration.md). The paired measurements
above retain their original source identity.
