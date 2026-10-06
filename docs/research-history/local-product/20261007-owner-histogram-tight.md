# 2026-10-07: tightly packed joint position histogram

Branch kernel/owner-histogram-tight based on PR50/c6ce329d. Prior three-histogram and padded-joint routes remain. Two new inline routes preserve the safe exact-coverage owner envelope while reducing redundant joint-bin padding. No mathematical factor,normalization,source/position-gradient,Polar update or snapshot changes.

Let T=max(input_count,output_count)+1. A current general-band atom at phase p=0,1,2 has joint_key=p*T+clipped_lo. Joint bins J=next_power_of_2(3T+1),nonmembers/padding useJ-1. Every active lookup p*T+t (0<=t<=direction_count) is below the sentinel; prior phases' counts remain exactly the prior-band counts. With s0 first general-band start,begin=s0+F(pT+clamp(j0-Lp+1)),end=s0+F(pT+clamp(j1)),the same safe envelope as PR48/50. No epsilon pruning,STE,global histogram/prefix/H or reduced position derivatives. Kernel static assertions enforce full range and sentinel capacity.

Full N64:old position stride128/joint512bins ->stride65/joint256. N128:256/1024 ->129/512. Sliced31/41 counts:stride42/joint128 instead64/256. Fewer joint histogram/cumsum bins may reduce register/shared work,but complete-step speed remains unverified. Inline copy uses2 direction CTAs for13 current fields; output/input owner16 sites,batch16 rows,atom chunk32,owner split4;parameter atom32/batch split2 oratom16/4warps/batch split2. Canonical IDs,fresh full-domain discrete-L2 normalizers and immutable snapshots retained.

N64/N128 B32 A204/A819 seed41,IEEE FP32/TF32off,initial performance rho>1 (1.25/3/8/mixed),production fused AdamW/Polar evolving widths,21samples/execution. Controls copy8,param2,atom16split2 and matching padded-joint routes. Dense is an independent nn.Linear performance reference,not equal initial matrix/update trajectory; initial Parameter hashes compare CST recipes only,input/target hashes include dense.

Host890 passed/922 skips(22.93s),changed-file ruff/diff,8 prepare/check pairs and isolated wheel/sdist build passed. New18GPU+1declaration suite covers full support/snapshot coverage,independent FP64 Y/dX/all canonical source gradients/nonzero positions,slices/B1/32/64,N64/N12820 captured reference optimizer updates/moments/steps,outstanding backward andempty-neighbor bands. GPU validation/timing pending. Raw owner-histogram-tight-20261007 evidence ignored/preserved.

Frozen source`3ae7a49d6823961c1e2f06aef9f2a8cda44ce326`; PR52(base PR50);queued check/N64/N128 `l4job-fc92633758f3456ab2c9f27693cf94d5`, `l4job-4cdcd271aece4b12a09d33e089130088`, `l4job-5bdef6df8ff340f9b07bcdd2a4cdd53d` on supervisor40251. No second GPU allocation.


## Retrieved GPU evidence

Analysis `screen` verifies 197 runtime files and source/result archive hashes. Full-shape FP64 checks cover Y, dX and all source-parameter gradients with nonzero position gradients. Initial decoded rho > 1 and forward widths change during production optimizer updates.

The table reports microseconds, median of execution medians. A screen has one execution per condition; two independent repeats are required for a repeatability claim. Dense uses a separately initialized weight matrix, with matched shape/input/target/dtype/optimizer and timing boundary; it is not the same initial CST operator. Full-domain discrete L2 normalization precedes slicing. Width VJP uses the existing fixed decoded-width source contract. No physical DRAM/cache residency claim follows from allocator peaks.

### local-histtight-64-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 59.24 | 115712 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 52.82 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.30 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 53.38 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 51.50 | 115712 | 6291456 |
| `local-ordered-reuse-histtightinline-param2-atom32` | 1 | 52.12 | 115712 | 6291456 |
| `local-ordered-reuse-histtightinline-paramatom16-param4-param2-atom32` | 1 | 50.21 | 115712 | 6291456 |
| `dense` | 1 | 39.09 | 34179584 | 48234496 |

### local-histtight-64-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 57.07 | 115712 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 53.02 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.16 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 53.09 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 51.47 | 115712 | 6291456 |
| `local-ordered-reuse-histtightinline-param2-atom32` | 1 | 52.13 | 115712 | 6291456 |
| `local-ordered-reuse-histtightinline-paramatom16-param4-param2-atom32` | 1 | 50.35 | 115712 | 6291456 |
| `dense` | 1 | 39.17 | 34179584 | 48234496 |

### local-histtight-128-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 70.55 | 303104 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 67.81 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 67.81 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 72.84 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 73.07 | 303104 | 6291456 |
| `local-ordered-reuse-histtightinline-param2-atom32` | 1 | 70.06 | 303104 | 6291456 |
| `local-ordered-reuse-histtightinline-paramatom16-param4-param2-atom32` | 1 | 70.13 | 303104 | 6291456 |
| `dense` | 1 | 45.87 | 34408960 | 48234496 |

### local-histtight-128-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 73.67 | 303104 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 69.54 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 69.92 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 75.27 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 75.47 | 303104 | 6291456 |
| `local-ordered-reuse-histtightinline-param2-atom32` | 1 | 72.09 | 303104 | 6291456 |
| `local-ordered-reuse-histtightinline-paramatom16-param4-param2-atom32` | 1 | 72.75 | 303104 | 6291456 |
| `dense` | 1 | 45.89 | 34408960 | 48234496 |


## Scoped independent repeat

Two queued all-rho jobs e5fe2c10/a5f293af were cancelled before execution to prioritize independent confirmation of the promising N64 rho3/8 screen and the G4 follow-up before 08:00. One selected repeat2 reuses three fixed CST routes plus dense, reverses rho/plan order, and is paired with the original screen. Only common selected routes have two executions; other screen routes have one. Narrow/mixed histogram timing remains unmeasured. This is deliberately scoped screening, not a full all-rho claim.

## Deadline and transport recovery

The selected reverse repeat87a13b88 was cancelled before execution after the preceding N128 reverse-result download failed and the pool safely stopped its owned L4. This candidate therefore has only one timing screen per tested rho3/8 condition, although GPU correctness19 tests passed. N64 approx50.21/50.35us is promising but not independently replicated; N128 approx70.06/72.09us is slower than matching uncached parameter controls. No all-rho or repeatability performance promotion. Remaining time was reserved for the already-validated G4 cached-atom16 follow-up.

## Completed short G4 independent comparison

After L419-test proof and promising N64 screening, frozen source3314fc57 ran a short G4 comparison. GPU suite19 passed (18 actual GPU + declaration,47.17s);197 committed/submitted/worker runtime hashes and source/result archives agree. Actual RTX PRO 6000 Blackwell Server Edition; decoded initial rho>1, evolving widths, FP32/TF32 off. Three selected CST routes plus independently initialized dense, two rho3/8 timing executions,21 graph samples each; repeat2 reverses rho/plan order. Twelve independent full-shape FP64 CST comparisons cover Y/dX/all source gradients/nonzero positions, and initial CST Parameter plus all-plan input bytes agree within-G4 across both repeats.

|N64 rho|copy8|uncached atom16/split2|tight histogram atom16/split2|dense|
|---|---:|---:|---:|---:|
|3|56.23|50.90|49.73|33.84|
|8|54.19|49.99|49.06|33.87|

Tight histogram beats the matched uncached parameter control in both executions/both rho conditions (1.17/0.93us aggregate,2.3/1.9%). Allocated peak115,712 B and CST reserved6,291,456 B unchanged; dense allocated34,179,584 B/reserved48,234,496 B includes workspace. It is1.47/1.45x the measured dense reference. This is a paired comparison within this source/G4 allocation, not a causal timing comparison to older G4 allocations. No narrow/mixed histogram or arbitrary-geometry performance claim. L4 histogram timing remains a single screen; the cancelled L4 reverse repeat does not become replicated by this G4 result.

Supervisor9465 drained exit0, owned sessiond778a6dc4fa2 stopped; lifecycle Session terminated/server No active sessions found and all slots stopped. No running/queued/interrupted pool work remains. All four G4 measurement artifacts now have byte-identical database export and idempotent reimport; family total8 artifacts (four L4 screens + four G4 selected repeats).
