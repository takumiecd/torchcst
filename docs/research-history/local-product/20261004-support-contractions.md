# Support-bounded local contractions

Research branch: `codex/local-product-hybrid`, parent `4de9ef4`.

This candidate removes dense input contractions for narrow support without
freezing sigma. Each forward freshly decodes the production polar activity state
and prepares normalized full-domain profiles, their center derivative correction
and exact positive-support intervals on the GPU. Slicing clips those intervals;
normalization remains over the full domain. No W or full U/V tensor is stored.

The prepared SoA is 13*A FP32 elements rather than 9*A: four additional rows hold
input/output starts and stops. Indices are exact integers <=128 encoded as FP32
for this bounded prototype. Empty supports clip to zero length.

For each atom block (initially 16), a uniform device-side branch selects:

- Both forward X->H and transposed dX use direct weighted reads from the current
  interval when its maximum local width in the block is <= the recipe limit.
- Parameter VJP uses interval contractions for H, dH, G and dG when both sides'
  maximum local widths are <= the limit.
- Wide/mixed blocks use the existing IEEE FP32 matrix contractions.

Limits 1/2/4/8 are recipe choices. This is a **group-level** adaptive contraction,
not a fully compacted per-atom dispatcher or shared/global-H reuse policy.
Forward H remains local, and output H->Y still uses the existing padded dot.
Intervals and branch decisions refresh inside CUDA Graph replay; no .item() or
host support decisions occur in a step. Task width derivatives remain detached;
production polar activity/regularization still updates sigma every training step.

The prototype reads narrow inputs by indices. Different centers can cause
noncontiguous lane accesses even though each atom's interval advances in order.
Packing remains a measured alternative; no claim that random access is eliminated
or that L1/L2 residency is guaranteed. Small dimensions alone do not establish a
performance improvement. Measure complete-step cost including interval preparation,
optional sorting, all gradients, AdamW proposal and polar policy update.

Research algorithm revision advances to `v2` because the explicit serialized
recipe now contains `support_limit` and supports the new route. Production
normalized algorithms remain at their unchanged revisions. To replay old `v1`
local-product snapshots, use checkpoint `4de9ef4` or its frozen source archive.

## Verification and reproducibility

Existing normalized cases still use the original runner/catalog. Use the same
runner for new candidate cases:

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-support.json \
  --case benchmarks/cuda/linear/cases/local-64-broad-support.json \
  --output benchmarks/cuda/linear/evidence/local-64-broad-support.json
```

CPU suite: 114 passed; 25 CUDA checks skipped locally. Additional CUDA cases cover
limits 1/2/4/8, spacings 1/.5, batch64, sliced 33/47 domains, empty/floor/live
singleton support, both narrow and fallback groups, and a graph transition from
narrow to wide support. Captured training also compares this route with public
CSTOptimizer's eager Y/dX/dP/parameters/AdamW state while sigma changes.

The first job `l4job-7ad307f5316740cd90d7a85c248b7289` (source SHA256
`0e6e1356aa2e3b5b00c98b35c4ae65bffb5573461e1ef26b62da3041056cd969`)
passed all 129 prior GPU checks but hit a Triton compile error in the new route:
reusing `_` for ignored values of different shapes introduced an incompatible
branch merge. The new route did not reach performance measurements. Ignored
returns now have distinct names. Frozen source and verified failure evidence are
preserved under ignored `benchmarks/cuda/linear/evidence/local-support-20261004/`.

## Verified negative result

Repaired job `l4job-f97128064c1346dd84e6f2a5b5fd04a7` passed **139 GPU tests**
and all five complete-step cases with plan-specific independent Y/dX/dP and
polar-update gates. Frozen source SHA256:

`21836831b60244cb6f06b84021dfdf62b51d62649cd125cb772bb23fd02bed97`.
Verified result archive SHA256:
`e146335af92e1f3516affd15b0b9024dc44a3c6df609897dc455d837b762c6b0`.
Raw source/driver/receipts/results remain in the ignored evidence job directory
and the shared pool. NVIDIA L4, driver580.82.07, 58 SM, Torch2.11.0+cu130,
CUDA13.0, Triton3.6.0, Python3.13.15; FP32/TF32-off; B32/N64/A205. Timings
include evolving sigma, preparation, forward, dX, all atom gradients, AdamW
proposal and production polar update. Seven synchronized graph replay samples,
separate workers in sequence; not paired timing.

| Initial profile | Existing unsorted (ms) | Support unsorted (ms) | Support packed (ms) | Dense (ms) |
| --- | ---: | ---: | ---: | ---: |
| aligned rho1 | 0.244893 | 0.262743 | 0.321793 | 0.039894 |
| rho2 | 0.244568 | 0.267214 | 0.325973 | 0.040176 |
| rho3 | 0.244483 | 0.272425 | 0.331432 | 0.039293 |
| rho16 | 0.245283 | 0.262620 | 0.322534 | 0.039181 |
| mixed | 0.245686 | 0.262871 | 0.327109 | 0.040929 |

All custom allocated peaks were 0.076660 MiB, reserved6 MiB, including graph
capture/replay; process usage unmeasured. The 4*A extra metadata is a tensor-size
calculation, not the measured peak. Torch factor peak33.362793 MiB and dense
32.596191 MiB, reserved46 MiB, in the same isolated-worker scope.

Compiler reports: support forward255 registers / dX254 vs existing168; no
reported spill slots. Support parameter VJP80 registers, shared16384 bytes vs
existing48/32768; prepare36 registers vs29. These are compiler reports, not an
occupancy/cache profile. No direct causal attribution is proven. Likely expenses
include branches, indexed reads, register pressure and repeated full-input reads
in fallback groups. Despite fewer narrow arithmetic operations, complete steps
were 7–11% slower. This route is retained as a negative research result, not a
new default. The next candidate reduces the unfused polar decode/autograd launch
chain while retaining the existing contractions.

Owned runtime stopped and slot verified stopped after retrieval.
