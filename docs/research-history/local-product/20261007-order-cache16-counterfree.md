# Count-free ordered cache with a 16-atom parameter tile

This combines PR #51’s exact cached ordering without diagnostic counters with PR #47’s 16-atom parameter tile, two batch partitions and four parameter warps. Eight CST controls/candidates and a separately initialized dense baseline cover N64/N128 and initial rho 1.25, 3, 8 and mixed; every decoded initial rho is strictly greater than one.

The runtime is unchanged from PR #51. Ordering repairs still use exact neighbor checks and full-sort fallback. Disabling counters removes their allocation and writes, not correctness checks. Position gradients, full-domain discrete L2 normalization before slicing, dX and all source-parameter gradients remain part of the contract; source-width VJP follows the existing fixed-decoded-width contract while optimizer steps change forward widths.

Host validation: 887 passed, 877 CUDA skips in 22.65 seconds, eight kernel comparison preparations/checks, wheel and source build. GPU correctness and performance are pending. No automatic dispatch or performance claim is introduced.


## Retrieved GPU evidence

Analysis `full-two-runs-partial` verifies 197 runtime files and source/result archive hashes. Full-shape FP64 checks cover Y, dX and all source-parameter gradients with nonzero position gradients. Initial decoded rho > 1 and forward widths change during production optimizer updates.

The table reports microseconds, median of execution medians. A screen has one execution per condition; two independent repeats are required for a repeatability claim. Dense uses a separately initialized weight matrix, with matched shape/input/target/dtype/optimizer and timing boundary; it is not the same initial CST operator. Full-domain discrete L2 normalization precedes slicing. Width VJP uses the existing fixed decoded-width source contract. No physical DRAM/cache residency claim follows from allocator peaks.

### local-cache16counterfree-64-rho1_25

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 55.50 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 50.04 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 1 | 49.37 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 1 | 49.52 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 1 | 49.54 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 49.23 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 49.18 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 49.48 | 116736 | 6291456 |
| `dense` | 1 | 38.93 | 34179584 | 48234496 |

### local-cache16counterfree-64-rho3

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 59.41 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.40 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 1 | 52.34 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 1 | 51.19 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 1 | 51.90 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 51.75 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 51.44 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 51.67 | 116736 | 6291456 |
| `dense` | 1 | 39.43 | 34179584 | 48234496 |

### local-cache16counterfree-64-rho8

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 56.82 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 51.44 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 1 | 50.51 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 1 | 50.35 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 1 | 50.66 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 50.46 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 50.34 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 50.55 | 116736 | 6291456 |
| `dense` | 1 | 39.20 | 34179584 | 48234496 |

### local-cache16counterfree-64-rhomixed

| Route | executions | step µs | peak allocated B | peak reserved B |
|---|---:|---:|---:|---:|
| `local-ordered-prep-copy8-atom32` | 1 | 63.00 | 115712 | 6291456 |
| `local-ordered-reuse-paramatom16-param4-param2-atom32` | 1 | 56.76 | 115712 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-copy8-atom32` | 1 | 57.01 | 117248 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-copy8-atom32` | 1 | 56.74 | 117248 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-copy8-atom32` | 1 | 56.66 | 117248 | 6291456 |
| `local-ordered-cache-gather-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 56.71 | 116736 | 6291456 |
| `local-ordered-cache-repair4-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 56.67 | 116736 | 6291456 |
| `local-ordered-cache-repair8-paramatom16-param4-param2-nostats-copy8-atom32` | 1 | 56.89 | 116736 | 6291456 |
| `dense` | 1 | 39.31 | 34179584 | 48234496 |

