# 2026-10-07: cache repair + parameter VJP, lower memory owner split

Branch kernel/small-linear-combined combines PR43 and PR42 on an isolated research branch. Existing recipes are retained. No main integration.

Three new cache routes combine current exact key-gather validation / bounded repair4 / repair8 with parameter atom32/batch split2. Every call still refreshes current coefficients, normalized support and independent autograd snapshots. Current canonical IDs retain exact gradient ownership; failed bounded repair performs full sort. New combined effects remain unmeasured.

One route keeps output/input owner16 dimensions, atom chunk32, parameter atom16/warps4/batch split2, and reduces owner split4 to split2. This halves the output/dX partial buffer capacity at the same tensor axes, but may lose GPU parallelism. Report actual complete-step peak and time rather than predicting peak from tensor budget. H is not allocated globally; physical DRAM/cache residency is not measured.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off; initial decoded rho>1 (1.25/3/8/mixed), production fused AdamW/Polar with evolving sigma. Controls copy8, param2 atom32 and atom16/warps4/batch2; dense same operator. 21 samples/execution, reverse case/plan order on repeat2. Independent FP64 Y/dX/all atom gradients and nonzero centers, singleton/norm-floor/empty support, slices/B1/32/64, old backward, N64/N128 20 captured reference optimizer updates/moments/steps.

Host:885 passed/814 GPU-or-resource skips (22.66s), changed-file ruff, isolated wheel/sdist build, 8 prepare/check passed. GPU checks and rho3/8 screening submitted only from committed snapshot. Ignored raw evidence under small-linear-combined-20261007.

Frozen source `aff57266b5eee6411a3391946fcbfbf07f66fb74`, draft PR44 (base PR43). GPU check `l4job-7a94bb7abf384ed08c1832d4aae8c8bc`, N64 rho3/8 screen `l4job-12801f3b5bf0458999aa319f5425c55c`, N128 rho3/8 screen `l4job-2b7811b4a8754c2a9b8d4916f0678465`. One L4 supervisor65855 owns this batch. Host CPU skips do not certify GPU success.

## Mathematical and task axes

For atom a, normalized input factor v_a and output factor u_a preserve the existing triweight/support/norm-floor contract. Y[b,j] = sum_a amplitude[a] u_a[j] sum_i X[b,i] v_a[i]. H[b,a] denotes the inner contraction, recomputed within owner blocks instead of stored globally. A singleton is determined by actual support, independently in each direction; rho alone is not a singleton criterion.

Forward/dX CTA: up to16 batch rows,16 output/input sites, atom record chunks32, four or two atom-list partitions per owner. This never means selecting16 dimensions from128 at random. Parameter VJP CTA: canonical atom records32 (cached routes) or16 (new low-memory route),16 batch rows,2 batch partitions. Each atom owns four canonical source parameter cotangents; fixed-source Polar pullback is linear in task cotangents, so sum of the two pulled-back contributions equals pullback of their sum up to FP32 rounding. All source, support and normalization derivatives are retained. Batch partials/reduction are included in complete-step measurement.

A single host orchestrator freezes source, queues complete independent experiments and retrieves/verifies archives; one L4 runs one complete job at a time. GPU CTAs are internally parallel and phases remain ordered. No parallel agents or competing GPU consumers are active.

Combined GPU suite passed64 tests (189.66s):36 new-route GPU checks +22 retained repair GPU checks +6 declarations/CPU boundaries. Source/result archives and committed/submitted/worker197 runtime files agree. Reference20 updates N64/N128 and nonzero positions retained. Raw proof test-proof.json preserved; measurement screens remain pending.

## L4 screening, 1 execution per condition

All28 full-shape FP64 plan comparisons and archive/runtime hashes passed, actual initialrho>1, source aff57266. Complete-step median us; allocated peaks include capture/replay. All reserved peaks6291456 bytes.

|N/rho|copy8|param2|atom16 split2/4warps|gather+param2|repair4+param2|repair8+param2|owner2 atom16split2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|64/3|58.89|52.38|50.90|53.08|52.24|52.94|55.83|39.30|
|64/8|56.58|52.42|51.12|52.35|52.13|52.02|54.92|39.28|
|128/3|70.13|67.89|67.90|68.73|69.51|66.20|69.31|45.69|
|128/8|73.88|69.40|69.80|66.27|66.08|65.97|79.64|45.69|

N64 peaks: controls115712/cache117248/owner2 99328 bytes; N128 controls303104/cache307200/owner2 270336 bytes. Owner2 reduces allocated peak16384/32768 bytes, but slows every screened condition vs best same-job candidate, especially N128/rho8. No universal win claim.

N128/rho3 repair8 changes full sorts forward49/dX50 to1/1 over52 primary refreshes, matching prior repair-only behavior. Repair4 leaves26/34 full sorts and is slower; repair cost plus fallback cannot be omitted. N128/rho8 gather+param2 and both repairs beat param2 even though repair4 still full-sorts28/27 times. Counter and separate-component diagnostics do not substitute for complete-step timing.

Full all8conditions/two independent executions queued from docs-only source1e1c8581 (same197 runtime files): `l4job-d1df360260b1426dad78bd6855942240` N64repeat1, `l4job-54a1ebc8a3164a2598a4aa2d83fb3eb3` N128repeat1, `l4job-cacdb986f5fb4bc083b424bf811f3455` N128repeat2, `l4job-b395c46d181c48fbaa566f98e221df99` N64repeat2. Seven CST routes+dense,21samples/execution, repeat2 reverses case/plan order. Supervisor65855 remains live with queued source-isolated prefix and batch-loop checks/screens before this full batch. CPU CI at head1e1c8581 SUCCESS.

N64 screen2 measurement artifacts DB byte-identical export/idempotent reimport passed. N128 screen2 artifactsもDB照合成功、screen合計4 artifacts保存。 The first direct-path ingest invocation lacked PYTHONPATH and failed before any DB action; reran with PYTHONPATH=. No evidence discarded.
