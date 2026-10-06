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
