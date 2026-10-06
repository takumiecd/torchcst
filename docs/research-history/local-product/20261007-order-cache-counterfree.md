# 2026-10-07: exact order cache without timed telemetry

Branch kernel/order-cache-counters-free based on PR44/ca14a8ab. Three new gather/repair4/repair8 parameter atom32/batch split2 routes disable order-cache telemetry. Existing counted routes retain their defaults. No exploration/normalization/support/derivative algorithm changes: current adjacent-key validation,bounded exact repair/full-sort fallback,fresh offsets/owner envelopes and13field snapshots/canonical gradient IDs remain.

Backend Recipe.record_order_stats defaults true. For new benchmark routes it is false; cache stats has shape(2,0),kernel Stats=None and CACHE_STATS=false removes diagnostic loads/stores at compile time. report returns counters_enabled=false,empty counts/names and explicit disabled scope,never fabricated zero execution counts. ID-cache allocation/order width remain unchanged. Counted routes still expose all original diagnostic counts. This isolates telemetry overhead and small persistent allocation, not a new exploration reduction. Peak allocator granularity may hide the logical48/80byte saving. Physical DRAM/cache residency unmeasured; no global H.

Axes: batch16 rows,output/input owner16 dimensions,consumer atom chunks32,owner split4;parameter canonical atom32,batch16/split2 and exact fixed-source Polar pullback+reduction. Initial performance rho>1 (1.25/3/8/mixed),N64/N128 B32 A204/A819 seed41,IEEE FP32/TF32off,production fused AdamW/Polar evolves widths,21samples/execution. Controls copy8,uncached atom16split2,matching three counted atom32 caches,dense.

Host886 passed/850 skips (22.00s), changed-file ruff,diff checks,8 prepare/check pairs and isolated wheel/sdist build passed. GPU correctness passed; retrieved timing evidence is below. New suite36GPU+1declaration: full FP64 Y/dX/all source gradients/nonzero positions,actual support/snapshot coverage,slices/B1/32/64,N64/N12820 captured reference optimizer updates/moments/steps,old backward/empty bands; additionally current cached vs independent full-sort snapshots under end/value changes,severely inverted cached permutation,old snapshot immutability,sliced geometry and zero-atom cache with genuinely absent counters. Retained counted repair tests are also included. Raw evidence order-cache-counterfree-20261007 ignored/preserved.

Frozen source426a42c1ebdce76b2e58583f6539f55f4e50ab04; GPU queue `l4job-78db53473279466bbd147c9bb56e5814`, `l4job-26739bfebaf54e27a3cc310d9020e489`, `l4job-fd03021ac392408cada1856ca75f8229` on L4 supervisor58836 after prior queued experiments. No other ownedGPU active.


## Retrieved GPU evidence

Analysis `retrieved-repeats` verifies 197 runtime files and source/result archive hashes. Full-shape FP64 checks cover Y, dX and all source-parameter gradients with nonzero position gradients. Initial decoded rho > 1 and forward widths change during production optimizer updates.

The table reports microseconds, median of execution medians. A screen has one execution per condition; two independent repeats are required for a repeatability claim. Dense uses a separately initialized weight matrix, with matched shape/input/target/dtype/optimizer and timing boundary; it is not the same initial CST operator. Full-domain discrete L2 normalization precedes slicing. Width VJP uses the existing fixed decoded-width source contract. No physical DRAM/cache residency claim follows from allocator peaks.

### local-cachecounterfree-128-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 70.54 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 68.01 | 303104 | 6291456 |
| `local-ordered-cache-gather-param2-copy8-atom32` | 2 | 69.11 | 307200 | 6291456 |
| `local-ordered-cache-repair4-param2-copy8-atom32` | 2 | 69.77 | 307200 | 6291456 |
| `local-ordered-cache-repair8-param2-copy8-atom32` | 2 | 66.36 | 307200 | 6291456 |
| `local-ordered-cache-gather-param2-nostats-copy8-atom32` | 2 | 68.64 | 306688 | 6291456 |
| `local-ordered-cache-repair4-param2-nostats-copy8-atom32` | 2 | 69.13 | 306688 | 6291456 |
| `local-ordered-cache-repair8-param2-nostats-copy8-atom32` | 2 | 66.27 | 306688 | 6291456 |
| `dense` | 2 | 46.02 | 34408960 | 48234496 |

### local-cachecounterfree-128-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 73.90 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 70.00 | 303104 | 6291456 |
| `local-ordered-cache-gather-param2-copy8-atom32` | 2 | 66.51 | 307200 | 6291456 |
| `local-ordered-cache-repair4-param2-copy8-atom32` | 2 | 66.57 | 307200 | 6291456 |
| `local-ordered-cache-repair8-param2-copy8-atom32` | 2 | 66.66 | 307200 | 6291456 |
| `local-ordered-cache-gather-param2-nostats-copy8-atom32` | 2 | 66.05 | 306688 | 6291456 |
| `local-ordered-cache-repair4-param2-nostats-copy8-atom32` | 2 | 66.34 | 306688 | 6291456 |
| `local-ordered-cache-repair8-param2-nostats-copy8-atom32` | 2 | 66.44 | 306688 | 6291456 |
| `dense` | 2 | 45.80 | 34408960 | 48234496 |

### local-cachecounterfree-128-rho1_25

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 66.01 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 64.36 | 303104 | 6291456 |
| `local-ordered-cache-gather-param2-copy8-atom32` | 1 | 60.93 | 307200 | 6291456 |
| `local-ordered-cache-repair4-param2-copy8-atom32` | 1 | 61.18 | 307200 | 6291456 |
| `local-ordered-cache-repair8-param2-copy8-atom32` | 1 | 60.91 | 307200 | 6291456 |
| `local-ordered-cache-gather-param2-nostats-copy8-atom32` | 1 | 60.66 | 306688 | 6291456 |
| `local-ordered-cache-repair4-param2-nostats-copy8-atom32` | 1 | 60.99 | 306688 | 6291456 |
| `local-ordered-cache-repair8-param2-nostats-copy8-atom32` | 1 | 60.65 | 306688 | 6291456 |
| `dense` | 1 | 45.66 | 34408960 | 48234496 |

### local-cachecounterfree-128-rhomixed

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 75.67 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 71.61 | 303104 | 6291456 |
| `local-ordered-cache-gather-param2-copy8-atom32` | 1 | 72.49 | 307200 | 6291456 |
| `local-ordered-cache-repair4-param2-copy8-atom32` | 1 | 69.25 | 307200 | 6291456 |
| `local-ordered-cache-repair8-param2-copy8-atom32` | 1 | 70.12 | 307200 | 6291456 |
| `local-ordered-cache-gather-param2-nostats-copy8-atom32` | 1 | 72.04 | 306688 | 6291456 |
| `local-ordered-cache-repair4-param2-nostats-copy8-atom32` | 1 | 69.51 | 306688 | 6291456 |
| `local-ordered-cache-repair8-param2-nostats-copy8-atom32` | 1 | 69.50 | 306688 | 6291456 |
| `dense` | 1 | 46.06 | 34408960 | 48234496 |


## Retrieval failure and repeatability scope

The intended reverse-order all-rho repeat2 (84e290c5) was interrupted on download: the transport log reports POOL_RESULT_READY but the requested remote archive was not found. There is no retrieved receipt/result proof, so it is excluded from performance/correctness counts. Supervisor40251 exited1, owned L4 stopped, lifecycle Session terminated/server No active sessions found and all slots stopped; pool recovery occurred only after terminal. No kernel failure is inferred.

The saved N128 rho3/8 screen and successful all-rho first run are independently retrieved and have identical initial CST Parameter/input bytes. These two same-order executions cover rho3/8 only. Narrow/mixed have one execution in this new no-counter study. Older counted combined PR44 has independently reversed all-rho repeats. The queued N64 tight histogram selected repeat87a13b88 was cancelled before execution after this transport failure to reserve the remaining time for the short already-validated G4 follow-up.

All eight counter-free measurement artifacts (four screens + four first all-rho records) have byte-identical database export and idempotent reimport. The archived screen JSON remains separately saved; the current summary explicitly includes per-route execution counts. No all-rho2 claim for N128 no-counter routes.
