# Subspacing widths and three-band H mixtures

Research branch `codex/local-product-hybrid`, parent `96e7f35d`.
The user asked to try sigma_min < spacing, dynamically dispatch narrow/medium/
wide atoms, and vary their proportions as in early/late training.

## Implementation and mathematical contract

The existing Linear benchmark now accepts an optional `widths` declaration:
shared minimum/birth/maximum, initial rho values and fractions, and optional
near-site center jitter. Defaults preserve all earlier cases. Local operator
validation recognizes the same product/Polar specification with these shared
bounds. Fixture initialization inverts the production activity map; sigma is
never frozen. The original 1024/8192 radial fixtures and public registry are
unchanged.

New research route `hybrid_three` takes boundaries `[1, mid, ...]`:

- rho <1: at most two positive input sites. Unroll two support accesses, with
  an actual-count guard. Retain full-domain normalized coefficients, including
  floor-active singletons; rho <1 is not generally a one-hot certificate.
- 1 <=rho <mid: contract the current positive input interval into local H.
- rho >=mid: produce H with the matrix contraction and store it for forward
  and parameter VJP reuse. Classification uses current precision every forward.

The two-access specialization also applies to the corresponding local input and
output contractions in parameter VJP. Transposed dX still uses the previous
matrix path. H-to-Y still uses a padded dot, canonical atom order is retained,
and saved H has fixed B*A capacity (only wide lanes are written/read). This
candidate does **not** yet implement output-owned packed scalar/few-site
accumulation or compact the wide H buffer. Cache residency is unmeasured.

Empty supports, full-domain-before-slice normalization, all parameter gradients,
detached task width and production Polar activity/radial updates are preserved.
New graph tests change all three bands through one captured graph, include
two-site rho=.75, floor-active singleton and empty support, and compare Y/dX/
all atom gradients with the independent FP64 scalar oracle. They also compare
the update against production. CPU configuration checks pass locally; CUDA
validation and timings are recorded below after verified result retrieval.

## Controlled proportions

N128/B32/A819 (~5% of dense entries), spacing1, minimum=birth=.25, maximum16,
FP32/IEEE, initial rho values [.25,.75,2,8], seed41. Centers are identical across
mixtures, within .1 of their nearest site, so the two subspacing widths are
initially live singletons. Amplitude/direction are matched across mixtures;
only radial activity and its width distribution change.

| Synthetic initial state | rho<1 | rho=2 | rho=8 |
| --- | ---: | ---: | ---: |
| early | 82 (10%) | 532 | 205 |
| middle | 410 (50%) | 286 | 123 |
| late | 778 (95%) | 33 | 8 |
| narrow-only | 819 (100%) | 0 | 0 |

These are **constructed initial width mixtures**, not proof that training
converges to narrow widths or a measurement of an actual trained model. All
measured steps perform real AdamW proposals and Polar activity updates.
Initial/final sigma and support counts are recorded outside timing. The narrow
fixtures must be reported separately from the ordinary sigma-three objective.

Compare mid=2/4/8 with prior two-band hybrid (mid4), all-local support H,
all-saved support H and all-saved matrix H. Same parameter/input hashes within
each case, separate subprocess per candidate. Warmup5, rounds21. Dense measured
once for the identical N/B/input/target/dtype/AdamW contract. The primary median
is the uninstrumented complete training-step CUDA Graph time. Allocated and
reserved peaks include capture/replay; whole-process GPU peak is unmeasured.
A separate event-instrumented graph provides phase diagnostics after these
primary measurements.

## Reproduction and evidence

```sh
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-mixture.json \
  --case benchmarks/cuda/linear/cases/local-128-mixture-late.json \
  --phase-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-mixture-late.json
```

Local checks: 133 passed,81 CUDA checks skipped. Ruff and whitespace checks pass.
Ignored raw driver/results are preserved in
`benchmarks/cuda/linear/evidence/local-mixture-20261004/` and the host-wide pool.
Job `l4job-53820428da27433e81b50cdb81a69a85`, frozen source SHA256
`97d7f05608bb464b697998324a5415eddc3e7c20b1afc71d12bb2fc3d43591ec`.

## Verified L4 results

Job succeeded; result archive SHA256
`463a3f7de4cd6537711192cfb4a92644d3e1678218551b8f3eabcea75b88bd60`.
NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0,
Python3.13.15. All214 L4-host checks pass in150.49s. All28 full N128 scalar
oracle comparisons and28 benchmark correctness gates pass. Maximum full-shape
absolute Y/dX/dP errors: 1.332e-6 / 1.559e-6 / 3.004e-6.
The owned runtime was stopped and all pool slots are stopped.

Complete-step uninstrumented CUDA Graph medians in milliseconds:

| Initial narrow share | Matrix saved | Support local | Support saved | Two-band mid4 | Three-band mid2 | Three-band mid4 | Three-band mid8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | .324704 | .392072 | .335107 | .422177 | .425847 | .425332 | .430332 |
| 50% | .325070 | .383041 | .330231 | .418422 | .420633 | .420705 | .424630 |
| 95% | .325383 | .336672 | .306622 | .400803 | .399187 | .399462 | .398353 |
| 100% | .325206 | .318940 | .295408 | .389480 | .382112 | .382143 | .382024 |

Narrow share improves support-aware arithmetic: 10% to100% reduces local time
by18.7%, saved support time by11.8%, and three-band mid4 time by10.2%.
Matrix saved H remains effectively flat. The all-local support route first
beats matrix-saved at the100% singleton fixture, but saved **support** H is still
faster. At10%, matrix-saved is fastest; at95/100%, support-saved is fastest.
This is evidence to tune contraction arithmetic as well as H lifetime.

The new three-band candidate remains slower than both saved alternatives.
Its two-access specialization improves the previous two-band route by about1.9%
at100% narrow support, but does not recover the hybrid overhead. The small
mid2/4/8 differences do not establish a universal cutoff; retain them as
configurable research variants. This is a negative result for this hybrid
schedule, not proof that efficient local atom records cannot win.

All819 atom sigmas changed in every candidate measurement. One-hot counts in
these near-site fixtures remained82/410/778/819. Routes also actually changed:
with mid2, the initial rho2 atoms cross into the local path (532/286/33 atoms in
the early/middle/late cases); with mid8, the rho8 atoms cross into local
(205/123/8 atoms). Thus the measured graphs follow current widths rather than
the initial labels. Initial assignments at thresholds remain subject to FP32
rounding; the CPU routing diagnostics state this limitation.

### Time attribution and memory

Separate event-instrumented phase medians (not primary step timings):

| Narrow share / route | Forward+loss ms | Backward ms | Optimizer ms |
| --- | ---: | ---: | ---: |
| 10% support local | .150432 | .147456 | .092160 |
| 100% support local | .115712 | .108544 | .092160 |
| 10% support saved | .092160 | .148480 | .092160 |
| 100% support saved | .091136 | .109568 | .092160 |
| 10% three-band mid4 | .186304 | .144384 | .092160 |
| 100% three-band mid4 | .149504 | .137216 | .092160 |

Hybrid's extra cost appears predominantly in forward; backward also retains
the width-insensitive matrix dX path. Current forward loops over atom blocks
inside two batch CTAs and accumulates the full padded output dot. The saved
producer has additional atom-block parallelism. Hybrid forward reports125
registers versus40 for support-local; three-band parameter VJP reports121
versus110 for the old hybrid. All compiler reports have zero spill slots.
These observations motivate output-owned packed accumulation and specialized
dX, but do not identify a measured cache-hit bottleneck or prove causality.

Measured peak allocation is invariant across these mixtures:

| Route | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: |
| Matrix saved | .252441 | 6 |
| Support local | .208496 | 6 |
| Support saved / all hybrids | .265137 | 6 |
| Dense | 32.814941 | 46 |

The ordinary dense complete step is .045613ms under the same N128/B32/FP32/
AdamW contract. Dense remains faster. No total-process GPU peak or L1/L2 cache
counters were measured. Do not interpret fixed B*A hybrid capacity as compact
wide-only storage. Raw samples and compiler reports are preserved in ignored
`summary.json`, with the verified job artifacts also retained on disk.

## Decision

Keep the three-band candidate and adjustable subspacing-mixture fixtures on the
research branch; do not promote it or set a hardcoded crossover. The requested
dynamic-width mixture comparison is complete. Narrow support does reduce time
in the tested support schedules, while this hybrid still needs better execution
layout. Prioritize output-owned packed scalar/few-site accumulation and narrow
dX, then repeat these same controlled mixtures. Preserve the saved matrix path
for broad arithmetic and the saved support path as a strong narrow control.
