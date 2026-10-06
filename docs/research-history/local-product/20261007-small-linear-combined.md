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

## Full repeat1 checkpoint (repeat2 pending)

Both size jobs succeeded;56 full-shape FP64 comparisons,197 runtime/archive proofs and within-case all-plan Parameter/input bytes incl dense passed. First independent full execution results only; repeat2 is in progress. Source1e1c8581.

|N/rho|copy8|param2|atom16split2/4warps|gather+param2|repair4+param2|repair8+param2|owner2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|64/1.25|55.00|51.04|49.72|49.88|50.15|50.44|51.75|39.17|
|64/3|58.98|52.50|51.45|53.08|52.47|53.11|55.82|39.18|
|64/8|56.41|52.64|51.09|52.16|52.07|51.96|54.77|39.21|
|64/mixed|62.69|58.01|56.69|57.50|57.60|57.90|56.38|39.21|
|128/1.25|65.96|64.74|64.46|60.96|60.98|60.79|65.34|45.65|
|128/3|70.34|67.78|67.83|68.54|69.44|66.19|68.85|45.69|
|128/8|74.10|69.04|70.02|66.06|66.07|66.31|79.47|45.55|
|128/mixed|75.74|71.26|71.56|72.17|69.62|69.73|74.15|45.50|

Median us,21samples/execution. Peaks follow screening: controls115712/303104,cache117248/307200,owner2 99328/270336 bytes;reserved6291456. N64 mixed owner2 now trades lower peak for time near/slightly below the speed-oriented candidate; this small difference needs repeat2. N128 mixed repair4/8 improves param2;rho3 repair4 remains worse. No single universal route or statistical significance claimed. All full artifacts remain raw/ignored; DB ingest is serialized by family.

G4 short selected comparison queued from docs-only source7e55e991 (same runtime/benchmark recipes as L4 validated source1e1c8581): check `colabjob-9e92c1692551429d86404e7a8e27fe7a` (host selectors collect27 cached+9 atom16split2 GPU tests),repeat1 `colabjob-634bd8f1c2024f21a0c6933ede97ddc0`,repeat2 `colabjob-224e457f27f64dcdb630304a66a63d68`. Only rho3/mixed N64/N128,copy8/uncached atom16split2 forced4/repair4+param2/repair8+param2+dense. Actual Blackwell/CC12.0 asserted; all within-G4 initial bytes/oracles preserved separately. G4 remains queued until L4 supervisor65855 drains and owned-stop/server confirmation; no concurrent G4 allocation.

## Completed independent L4 repeat1 + repeat2

All four full jobs succeeded. Source1e1c8581, 112 full-shape FP64 comparisons, matching committed/submitted/worker197 runtime files and archive SHA256, all-plan initial Parameter/input bytes including dense and both executions. Actual initial rho>1; all width updates evolve. Each cell is the median of two independent execution medians (21 samples each); repeat2 reverses case/plan order. This supersedes the pending repeat2 checkpoint above. All16 full measurement artifacts and4 screen artifacts have byte-identical DB export and idempotent reimport.

|N/rho|copy8|param2|atom16split2/4warps|gather+param2|repair4+param2|repair8+param2|owner2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|64/1.25|54.89|51.09|49.88|49.91|50.30|50.33|51.80|39.14|
|64/3|59.07|52.50|51.36|52.96|52.48|52.95|55.68|39.30|
|64/8|56.44|52.61|51.14|52.29|52.01|52.02|54.81|39.11|
|64/mixed|62.55|57.98|56.63|57.78|57.65|58.07|56.15|39.21|
|128/1.25|65.96|64.65|64.21|60.91|60.95|60.76|65.37|45.75|
|128/3|70.13|67.66|67.72|68.51|69.54|66.12|69.04|45.59|
|128/8|73.86|69.10|69.73|66.20|66.18|66.34|79.35|45.49|
|128/mixed|75.95|71.18|71.44|72.28|69.31|69.95|74.15|45.37|

N64 atom16/split2 improves copy8 in all four conditions in both executions. N64 mixed owner2 has lower allocated peak99,328 vs115,712 bytes and a small median-time advantage56.15 vs56.63 us; both independent runs have this direction, but this is not a statistical-significance claim. N128 rho1.25/3/8/mixed has useful cache routes; repair8 is strongest at rho3, repair4 at mixed. These exact fixed routes were measured; no dynamic dispatcher or measured universal default is claimed. N128 wide owner2 is a negative result79.35 us.

N128 rho3 full sort counts over52 primary refreshes: gather49/50, repair4 forward/dX26/34, repair8 1/1 in both runs. At rho8 repair8 reduces full sorts to4/3 versus gather28/28, yet the complete time remains similar to gather and repair4. Fewer full sorts alone is insufficient; repair, current-value refresh and parameter gradients remain material. Controls peak115,712/303,104 bytes, caches117,248/307,200, owner2 99,328/270,336; all reserved6,291,456. No new global H; DRAM traffic or L2 residency is not established by these measurements.

## Owned runtime recovery and G4 start

After successful inline-histogram test retrieval, L4 supervisor65855 exited1 on job cleanup transport timeout(180s). All preceding successful measurements/checks and source/archive proofs remain valid; no kernel failure is inferred. Pool-owned L4 sessioncst-pool-8efd3105f08a-1 stopped with lifecycle Session terminated and server No active sessions found; all slots stopped and pool recover after terminal returned0. Remaining L4 inline screens,cached-atom16 repeats and joint-histogram experiments remain queued.

At21:35UTC short already-queued G4 comparison started first on supervisor21298,one owned sessioncst-pool-156f0fde07df-1,endpointgpu-g4-s-ft-kkb-euw4a1-2ubn8by91reyk. No concurrent L4 or other ownedGPU allocation. After G4 drains and owned-stop/server verification, resume remaining L4 queue before08:00JST endpoint.

The earlier independent G4 param2 reference-update proof from PR38 has been copied as docs-only commit869d6bb1 to preserve up-to-date inherited records. Combined runtime/benchmark measurement source remains unchanged.
