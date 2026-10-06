# Count-free ordered cache with a 16-atom parameter tile

This combines PR #51’s exact cached ordering without diagnostic counters with PR #47’s 16-atom parameter tile, two batch partitions and four parameter warps. Eight CST controls/candidates and a separately initialized dense baseline cover N64/N128 and initial rho 1.25, 3, 8 and mixed; every decoded initial rho is strictly greater than one.

The runtime is unchanged from PR #51. Ordering repairs still use exact neighbor checks and full-sort fallback. Disabling counters removes their allocation and writes, not correctness checks. Position gradients, full-domain discrete L2 normalization before slicing, dX and all source-parameter gradients remain part of the contract; source-width VJP follows the existing fixed-decoded-width contract while optimizer steps change forward widths.

Host validation: 887 passed, 877 CUDA skips in 22.65 seconds, eight kernel comparison preparations/checks, wheel and source build. GPU correctness passed; retrieved timing evidence is below. No automatic dispatch or performance claim is introduced.


## Retrieved GPU evidence

Analysis `full-two-runs` verifies 197 runtime files and source/result archive hashes. Full-shape FP64 checks cover Y, dX and all source-parameter gradients with nonzero position gradients. Initial decoded rho > 1 and forward widths change during production optimizer updates.

The table reports microseconds, median of execution medians. A screen has one execution per condition; two independent repeats are required for a repeatability claim. Dense uses a separately initialized weight matrix, with matched shape/input/target/dtype/optimizer and timing boundary; it is not the same initial CST operator. Full-domain discrete L2 normalization precedes slicing. Width VJP uses the existing fixed decoded-width source contract. No physical DRAM/cache residency claim follows from allocator peaks.

### local-cache16counterfree-64-rho1_25

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 55.49 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 50.01 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 2 | 49.33 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 2 | 49.50 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 2 | 49.59 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 49.10 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 49.26 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 49.50 | 116736 | 6291456 |
| `dense` | 2 | 39.14 | 34179584 | 48234496 |

### local-cache16counterfree-64-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 59.39 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 51.48 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 2 | 52.12 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 2 | 51.31 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 2 | 51.93 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 51.52 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 51.20 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 51.79 | 116736 | 6291456 |
| `dense` | 2 | 39.32 | 34179584 | 48234496 |

### local-cache16counterfree-64-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 56.88 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 51.58 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 2 | 50.58 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 2 | 50.49 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 2 | 50.65 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 50.47 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 50.41 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 50.52 | 116736 | 6291456 |
| `dense` | 2 | 39.24 | 34179584 | 48234496 |

### local-cache16counterfree-64-rhomixed

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 2 | 63.04 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 2 | 56.81 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 2 | 57.09 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 2 | 56.94 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 2 | 56.75 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 56.72 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 56.61 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 2 | 56.68 | 116736 | 6291456 |
| `dense` | 2 | 39.28 | 34179584 | 48234496 |



## Independent repeat interpretation

Two independent all-rho N64 executions completed: 64 full-shape CST FP64 comparisons, 197 committed/submitted/worker runtime file hashes, initial decoded rho >1, evolving forward widths and within-case/repeat initial CST Parameter bytes plus all-plan input/target equality. GPU route suite: 28 passed (27 GPU + declaration).

Counter-free cache peaks are116,736 B vs117,248 B counted (512 B lower); uncached remains115,712 B. CST reserved remains6,291,456 B. Dense allocated is34,179,584 B and reserved48,234,496 B including workspace.

Narrow gather/nostats49.10 us vsuncached50.01; wide repair4/nostats50.41 vsuncached51.58. Both directions hold in both runs. Middle repair4/nostats51.20 aggregate vsuncached51.48, but is slower in repeat1 and faster in repeat2: not a stable win. Mixed repair4/nostats56.61 vsuncached56.81 is faster in both runs but only0.20us aggregate; no significance claim.

Matched no-counter vs counted cache route/regime comparisons are faster in both executions for10/12 pairs; middle repair4 and mixed repair8 change direction. This is a small telemetry/payload improvement, not an exploration algorithm change.
