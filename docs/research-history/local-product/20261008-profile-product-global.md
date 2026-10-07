# Single Product chart: whole-chart H/G at1024²

## Contract and implementation

This work follows the input-Strip grid in PR #61. It adds the explicit research
Algorithm `research_profile_product_global/v1`, with one regular Euclidean
Product chart. Both output and input dimensions may be2..1024, batch1..64,
up to65536 atoms, FP32/IEEE, equally spaced Lines, and Polar profile_product
with raw Triweight factors. The public dispatcher is unchanged. Chart gradients
and higher-order derivatives are outside this CUDA core's support.

For atom a, the regular coordinates are o_i=O_o+iS and x_j=O_i+jS.
Let u_a(i), v_a(j) be its raw profiles and define

\[
D_a=\max\left(\|u_a\|_2\|v_a\|_2,\varepsilon\right),\qquad
W_{ij}=\sum_a\mathrm{amp}_a\frac{u_a(i)v_a(j)}{D_a}.
\]

Normalization uses the complete Product chart. Above the single global floor,
use U=u/||u|| and V=v/||v||. Below it, split the denominator with sqrt(epsilon)
on each factor. Thus U V=u v/D in both cases; separately flooring each axis
would change the mathematical contract. The task width derivative is detached,
while the production Polar update evolves widths during every optimizer step.

The contractions preserve the small-core idea:

\[
H_{ba}=\sum_jX_{bj}V_a(j),\quad
Y_{bi}=\sum_a\mathrm{amp}_a H_{ba}U_a(i),
\]
\[
G_{ba}=\sum_i\overline Y_{bi}U_a(i),\quad
\overline X_{bj}=\sum_a\mathrm{amp}_a G_{ba}V_a(j).
\]

H/G are O(BA); two support-ordered metadata views are O(A). No full W/dW is
materialized. Center VJPs include complete-domain normalization derivatives;
global singleton certificates suppress exact zero center derivatives only above
the product floor. Canonical atom IDs retain gradient identity after sorting.

The shared executor/kernels now live in `profile_product_global`; the Strip
grid wrapper supplies its physical pitch/tile coordinates and keeps its prior
full preparation and launch settings. Each forward snapshots source parameters
and amplitude once; Strip also snapshots live pitch. Older VJPs retain their own
values even after another forward or live state mutation.

Two new regular-Product preparation recipes share the same contractions:

- `grid-full`: full-axis normalization preparation, inherited from the small core.
- `grid-support`: conservative support spans, dynamic32-site scans, complete norm
  and derivative sums over every nonzero site. Numerically uncertain spans fall
  back to the full axis. There is no fixed support capacity or truncation.

For A>4096, key generation and sorting avoid one huge Triton sort CTA. Chunk
maxima of support ends and a lower-bound search on sorted starts conservatively
bound each owner range. Ends need not be monotonic: an early wide interval that
crosses many later owners is retained. Owners may scan holes inside these bounds;
the profile itself supplies exact zero masks. Atom ordering and current supports
are rebuilt on every forward/replay, including after width/center updates.

## Verification

- CPU:1001 passed,1236 skipped; skipped GPU/DB tests are not validated by this run.
- Actual NVIDIA L4:30 new Product tests passed, including19 CUDA tests;56 existing
  Strip tests also passed, including45 CUDA tests. Cases cover all-site Y/dX/all
  canonical gradients, strided tensors, batch64, A4097, empty/singleton/floor,
  width400, nonmonotonic ends, old-forward snapshots and20 captured optimizer
  updates including parameters, moments, clocks and evolving widths.
- Wheel/sdist build and wheel metadata import without Triton passed. Runtime
  metadata does not import benchmark tooling or GPU kernels.
- The benchmark oracle independently constructs FP64 Polar widths/amplitudes,
  raw factors over all axis sites and the factored complete Product norm. Only
  atom count is chunked. It checks every Y/dX/atom gradient at the actual large
  shapes. CPU tests compare this factorization to an independently normalized
  raw matrix oracle. Adapter revision6 states this distinct scope explicitly;
  it is not the earlier Strip raw matrix oracle at the large shapes.
- Eleven complete runs pass adapter revision6 and public submission consistency
  checks after bounded diagnostic export. Maximum absolute errors across all
  routes/runs are Y1.8725e-5, dX1.7298e-5 and dp4.6085e-6; the production
  Polar update with the same supplied cotangent matches exactly.
- Y/dX/dp tolerance remains4e-4; production Polar update with the same supplied
  cotangent remains2e-6. No tolerance was relaxed. Actual DB ingestion and
  production dispatcher adoption are not part of this record.

## Complete-step results

NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. Single Product chart,
square256/512/1024, B32, A3277/13107/52429 (~5%), seed41. Widths start at rho3
or rho8 and evolve. Fused capturable AdamW (lr1e-4, weight_decay.01) plus the
production fused Polar update are inside every measured step. Each number is
the median of21 synchronized, uninstrumented complete CUDA Graph steps;
timing workers use separate processes. No samples or unfavorable comparisons
are discarded. Dense uses FP32/IEEE and the same input/target, with its ordinary
AdamW update; it is a performance reference with a different parameterization.

| size / initial rho | Torch ms | grid-full ms | grid-support ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
|256² /3|0.59966|0.23577|0.23106|0.06200|
|256² /8|0.60325|0.23983|0.23380|0.06157|
|512² /3|10.75108|0.49648|0.47177|0.06247|
|512² /8|10.79172|0.51010|0.48560|0.06237|
|1024² /3|121.51867|1.90825|1.67720|0.08255|
|1024² /8|121.71797|1.99566|1.72871|0.08270|

At rho3, grid-support is2.60x/22.79x/72.45x faster than the factored Torch
baseline. Exact support preparation improves over grid-full by2.0%/5.0%/12.1%.
The remaining gap versus dense grows from3.73x to7.55x to20.32x: success against
Torch does not establish dense competitiveness. This is the first bounded
large-Product implementation, not a launch-parameter sweep or8192² result.

| size, rho3 | Torch allocated / reserved bytes | grid-support allocated / reserved bytes | dense allocated / reserved bytes |
| --- | ---: | ---: | ---: |
|256²|86558208 /171966464|1814016 /6291456|35260928 /48234496|
|512²|444653056 /866123776|6674944 /50331648|38537728 /52428800|
|1024²|3261389312 /6582960128|25604096 /102760448|51382784 /111149056|

These are isolated-process warmed peaks including capture/replay, gradients,
optimizer state and framework workspaces; reserved is separate from allocated.
GPU process memory was not measured. Dense has a substantial pre-capture
allocation whose cause is not instrumented; this table is not a model-only
memory comparison. Both grid preparation recipes have the same measured peaks.

Reverse-order rho3 runs give grid-support0.22997ms at256² and1.61609ms at1024².
Another1024² reverse-order run with phase diagnostics after its primary timing
gives1.73702ms (primary uninstrumented timing). After the diagnostic-export fix,
an independent1024² run gives1.57763ms/rho3 and1.65592ms/rho8. All results are
retained:1024² rho3 spans1.57763..1.73702ms across four independent complete
runs; the primary comparison table is not replaced with the fastest sample.

The separate instrumented1024² rho3 graph reports grid-support forward/loss
0.74240ms, backward0.80077ms and optimizer0.04403ms. Its old-point snapshot,
AdamW and Polar component medians are0.00410/0.02867/0.00717ms. These external
event diagnostics run after the main measurement with continuing width updates;
their sums are not a substitute for the uninstrumented complete-step time.
Forward/backward dominate, while preparation alone saves only12% in the primary
case. Per-kernel sorting, H/G access and contraction costs have not yet been
isolated, so no particular kernel is claimed to be the next proven bottleneck.

## Current old Strip/Torus control

The current public `cuda_strip_torus_fused/v1` was also measured atoutput64,
input256/512/1024, B32, A819/1638/3276, seed41. X and target hashes match the
earlier Polar input-Strip benchmark. Every site and canonical gradient passes
an atom-chunked FP64 public Torch radial reference independent of CUDA;
maximum errors are Y3.33e-6, dX2.28e-10 and dp5.08e-7. Every atom's width
changes across26 complete optimizer steps and all parameters remain finite.

| input | old forward+backward Graph ms | old complete eager ms | Polar input-Strip grid complete eager ms |
| --- | ---: | ---: | ---: |
|256|0.47389|9.77349|2.19240|
|512|1.21034|9.81635|2.12783|
|1024|4.02956|11.31152|2.17893|

The old complete eager path includes the public CSTOptimizer's gradient
projection, fused AdamW, Direct activity width update and Torus moment transport.
General Torus coordinate update explicitly does not support Graph capture; its
Graph column therefore excludes optimizer. The Polar benchmark uses its fused
update binding rather than this eager public wrapper. The declarations also
differ: old output Strip16/Torus, raw radial profile without L2, Direct widths
about0.5 with input Grid spacing0.05; new input Strip64/Euclidean Lines,
normalized profile product, Polar widths initialized at3 with spacing1.
This is a shape/input-matched control, not an equal-math speedup comparison.
The previously archived faster bounded-list/FP16-weight-cache prototype is
separate from this current runtime and is not ported or remeasured here.

## Bounded diagnostic export

The first1024² combined JSON was9.6..9.8MiB because three routes each embedded
initial/final widths for52429 atoms. Its full worker JSON and original combined
artifact are preserved. Consolidated Product results now retain exact range,
changed count, maximum change and SHA256 of each exact compact-JSON width array;
the full arrays remain in the unmodified measure worker JSON. Older fixtures keep
their original format. This affects diagnostic serialization only, after GPU
measurement; all timings, correctness metrics, source provenance, execution UUID,
inputs and snapshot identities are unchanged.

Earlier artifacts were exported with this same helper; original/export hashes
are both recorded. The final GPU run generated the bounded format directly
(~263KB per1024² result) and passed all correctness workers and adapter checks.
CPU regression verifies the52K-atom export fits the unchanged5MiB submission
limit, preserves source records and hashes, and does not alter measured values.
No file-size or numerical acceptance limit was increased.

Two old-control driver attempts failed before producing a valid result: one
imported the input helper from the wrong module; another called an unavailable
width diagnostic facade after the first F/B capture. Both source archives/logs
are retained. After correcting those diagnostic references and checking them on
CPU, the explicit new old-control job passed. None of the failed attempt timings
is used. Three queued global jobs were canceled before execution after fixing
the driver's catalog path; they were not GPU measurements.

## Reproduction and evidence

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-global.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-256-rho3.json \
  --candidate grid-full --candidate grid-support --output output/global-comparison
python -m tools.kernel_dev check \
  --plans output/global-comparison/plans.json \
  --case output/global-comparison/case.json
python -m tools.kernel_dev test --suite profile-product-global
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-profile-product-global.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-1024-rho3.json \
  --polar-update fused --source-commit COMMIT --output output/global-result.json
python -m benchmarks.submissions check output/global-result.json
```

Measured runtime source:6803dad490fd094e51d420d00930237149a5c94c. New Product
and Strip regression test job:l4job-3f24b7e157984d4aa4d959e5d0616cce. Primary
measurement jobs:l4job-6b3f6225fdab427188125024dd47a348 (256),
l4job-50298d74c019445bb002e482e0591573 (512),
l4job-4a3856b12d0940dfa56d1066458f2619 (1024).
Reverse jobs:l4job-f4fac95b66fc446e8a5a418dd4bc7cb8 (256),
l4job-d3fe74f688694cdabcb3aecc78991b48 (1024),
l4job-77b6f933a38f4e918d8b906bd313a2f4 (1024 with post-measurement phases).
Final bounded-export source:fcf0e3ed8bc298770892341de53266c70741957a;
job:l4job-f23cb70e2c6f461f8131934e961bca34. Current old Strip/Torus control:
l4job-9078a957a7e84c3fa36fdf0163ab7a1a.

Raw source archives, drivers, logs, result JSON, pool receipts and test/build
output are preserved in ignored `benchmarks/cuda/linear/evidence/profile-product-global-20261008/`
and the default shared-pool job directories. This worktree is retained. Tracked
summary JSON contains result/source/archive hashes rather than raw evidence.
