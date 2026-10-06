# 2026-10-07: one joint position histogram with inline metadata copy

Branch kernel/owner-histogram-fused based on PR49. Preserve all prior routes; two new inline routes use the same parameter atom32/batch split2 or atom16/4warps/batch split2. Current triweight, normalization, canonical IDs, position/all source gradients and per-forward snapshots remain exact.

For current nonsingleton band p=0,1,2 and relative integer support start l_a, set joint_key=p*P+l_a, P=next_power_of_2(max(N,K)+1). One histogram of4P bins counts all active band atoms; nonmembers/padding use4P-1, outside every queried index. Its prefix F(pP+t) includes all prior bands and current-band starts<t. With s0 the first nonsingleton band start, owner[j0,j1) uses begin=s0+F(pP+clamp(j0-Lp+1)), end=s0+F(pP+clamp(j1)), Lp=max current span of band p. This is algebraically the same safe envelope as PR48; there is no per-band prefix subtraction or extra global scratch. Empty bands and singleton records retain current membership semantics. The sort/current refresh remain; three per-band histograms/prefixes become one larger histogram/prefix, which may increase shared/register cost and is not presumed faster. Actual singleton flags are independent of rho.

Inline copy remains2 direction CTAs for13 current physical metadata fields and safe owner envelopes, skipping a second copy/range launch. Forward/dX owner16 dimensions,16 batch rows,atom chunk32,owner split4; parameter canonical atom16 or32,16 batch rows/split2. H recomputed locally, no global H or joint histogram tensor. Physical DRAM/L2 residency unmeasured.

N64/N128 B32 A204/A819 seed41,IEEE FP32/TF32off,actual initialrho>1 (1.25/3/8/mixed),production fused AdamW/Polar evolves widths;21 samples/execution. Controls copy8,param2,atom16split2,matching inline3-histogram routes anddense. GPU correctness suite18 tests plus1 declaration covers full support/snapshot coverage, independent FP64 values/dX/all source gradients/nonzero positions,slices/B1/32/64,N64/N12820 captured reference updates/moments/steps,outstanding old backward and empty neighbors. GPU proof and timings pending.

Host889 passed/904 skipped (22.82s), changed-file ruff,diff checks,8 successful prepare/check pairs and isolated wheel/sdist build. Initial prepare helper used an invalid positional check argument and failed after first safe preparation; preserved that output/log and reran all8 with named --plans/--case in a fresh output directory. No overwrites. CPU skips are not GPU certification. Raw evidence owner-histogram-fused-20261007 ignored and preserved.

Frozen source10ed2a79; check `l4job-8a104834ceb04733a877f3dd96d43a07`, N64screen `l4job-457d33b38c914f89b27329eebe5d0e31`, N128screen `l4job-1f18ada6fdba4366b39a2394a6c37b48` queued on existing L4 supervisor65855 after cached-atom16 full repeats. No competing GPU allocated.


## Retrieved GPU evidence

Analysis `screen` verifies 197 runtime files and source/result archive hashes. Full-shape FP64 checks cover Y, dX and all source-parameter gradients with nonzero position gradients. Initial decoded rho > 1 and forward widths change during production optimizer updates.

The table reports microseconds, median of execution medians. A screen has one execution per condition; two independent repeats are required for a repeatability claim. Dense uses a separately initialized weight matrix, with matched shape/input/target/dtype/optimizer and timing boundary; it is not the same initial CST operator. Full-domain discrete L2 normalization precedes slicing. Width VJP uses the existing fixed decoded-width source contract. No physical DRAM/cache residency claim follows from allocator peaks.

### local-histfused-64-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 59.57 | 115712 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 52.89 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.38 | 115712 | 6291456 |
| `local-ordered-reuse-histinline-param2-atom32` | 1 | 52.35 | 115712 | 6291456 |
| `local-ordered-reuse-histinline-paramatom16-param4-param2-atom32` | 1 | 51.09 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 52.78 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 51.54 | 115712 | 6291456 |
| `dense` | 1 | 39.33 | 34179584 | 48234496 |

### local-histfused-64-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 57.02 | 115712 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 52.95 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.63 | 115712 | 6291456 |
| `local-ordered-reuse-histinline-param2-atom32` | 1 | 52.36 | 115712 | 6291456 |
| `local-ordered-reuse-histinline-paramatom16-param4-param2-atom32` | 1 | 50.83 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 52.99 | 115712 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 51.82 | 115712 | 6291456 |
| `dense` | 1 | 39.18 | 34179584 | 48234496 |

### local-histfused-128-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 70.14 | 303104 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 68.10 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 67.86 | 303104 | 6291456 |
| `local-ordered-reuse-histinline-param2-atom32` | 1 | 70.11 | 303104 | 6291456 |
| `local-ordered-reuse-histinline-paramatom16-param4-param2-atom32` | 1 | 70.09 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 73.27 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 73.12 | 303104 | 6291456 |
| `dense` | 1 | 45.84 | 34408960 | 48234496 |

### local-histfused-128-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 73.96 | 303104 | 6291456 |
| `local-ordered-reuse-param2-atom32` | 1 | 69.78 | 303104 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 69.95 | 303104 | 6291456 |
| `local-ordered-reuse-histinline-param2-atom32` | 1 | 72.13 | 303104 | 6291456 |
| `local-ordered-reuse-histinline-paramatom16-param4-param2-atom32` | 1 | 71.98 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-param2-atom32` | 1 | 75.61 | 303104 | 6291456 |
| `local-ordered-reuse-histfusedinline-paramatom16-param4-param2-atom32` | 1 | 75.68 | 303104 | 6291456 |
| `dense` | 1 | 46.04 | 34408960 | 48234496 |

