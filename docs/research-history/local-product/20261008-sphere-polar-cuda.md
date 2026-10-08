# Existing Sphere Polar CUDA contractions

2026-10-08. This experiment optimizes the existing two explicit intrinsic S²
charts and separable Polar/Triweight kernel. Each profile uses the complete
chart's discrete L2 norm with its own floor. The geometry remains embedded
chord distance and the task VJP detaches widths; public CSTOptimizer retains
activity, regularization, Sphere retraction and state transport.

## Mathematical and execution contracts

For normalized profiles v(i,a), u(o,a), amplitude w(a),

\[
H_{ba}=\sum_i X_{bi}v_{ia},\quad
G_{ba}=\sum_o dY_{bo}u_{oa},\quad
Y_{bo}=\sum_a w_aH_{ba}u_{oa}.
\]

The amplitude VJP is \(\sum_b H_{ba}G_{ba}\). For either profile's
raw-site VJP, the normalization correction has projection
\(w_a\sum_bH_{ba}G_{ba}\), except below that profile's norm floor.
Ambient center VJPs are reduced first, then pulled back through the intrinsic
Sphere exponential map once per atom. Direct three-component offsets avoid
subtracting nearly equal radius-squared dot products.

- Bounded factors: evaluate complete profiles for an atom chunk in CUDA,
  contract using IEEE Torch/cuBLAS GEMMs, retain compact H or recompute it,
  and regenerate temporary factors in backward. No full sites-by-all-atoms
  geometry array is saved.
- Support contractions: retain complete-support indices, normalized profiles
  and H in bounded buffers, and perform atom-wise contractions. A support
  count greater than capacity uses exact full-site traversal; no contribution
  is truncated.
- Weight contractions: assemble W from support patches, run Y/dX/dW with
  Torch/cuBLAS, and compute every atom's VJP against shared dW. W and dW are
  ordinary square arrays, without the atom-by-output-by-input tensor.

Each forward snapshots source parameters, geometry, normalization and Polar
quantities for its backward. Chart gradients and higher derivatives are out
of scope. Metadata restricts execution to CUDA FP32 IEEE, two fixed explicit
intrinsic S² charts, 1..2048 sites per chart and batch 1..64, revision1 Triweight and the existing1e-6 profile floor. Atomic support
and weight plans reject deterministic mode. Algorithms are registered only
in the benchmark registry; public selection is unchanged.

## Primary measurements

NVIDIA L4, Torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0. N inputs and outputs,
B=32, A=floor(0.05*N²), initial width 3 or 8, uniform random surface sites,
radius sqrt(N/(4*pi)), initial seed41. This radius is a fixture choice.
Uninstrumented complete eager steps include dX, fused AdamW and the public
geometry/activity update, with three warmups and 21 timed evolving steps.

| N | width | route | complete step ms | allocated MiB | reserved MiB |
|---:|---:|---|---:|---:|---:|
| 1024 | 3 | original factored | 168.860 | 4120.127 | 4832 |
| 1024 | 3 | bounded 4096 + H | 19.264 | 131.933 | 146 |
| 1024 | 3 | support64 + H | 14.602 | 87.012 | 112 |
| 1024 | 3 | support64 + W | 13.276 | 88.486 | 112 |
| 1024 | 8 | bounded 4096 + H | 18.331 | 131.933 | 146 |
| 1024 | 8 | support256 + H | 49.117 | 243.813 | 258 |
| 1024 | 8 | support256 + W | 146.518 | 245.287 | 258 |
| 2048 | 3 | bounded16384 + H | 108.311 | 863.395 | 998 |
| 2048 | 3 | support64 + H | 49.627 | 303.559 | 332 |
| 2048 | 3 | support64 + W | 38.040 | 308.008 | 338 |
| 2048 | 8 | bounded16384 + H | 108.520 | 863.395 | 998 |
| 2048 | 8 | support256 + H | 216.830 | 914.758 | 948 |
| 2048 | 8 | support256 + W | 595.488 | 919.208 | 954 |

These rows are one independent primary run each, not 21 independent runs.
The original 2048 factored route's full-site OOM is preserved in the previous
Sphere baseline record. All new 2048 routes passed full-shape correctness
before and after the 24 optimizer steps. The dense reference remains faster
(~0.72ms at1024 and~1.13ms at2048), with a different parameterized model.

Width-three favors support-patch W. Width-eight favors bounded H/G
contractions; the support-patch product grows quadratically with support
size. W256 is a negative tuning result, not a candidate for default dispatch.
The support64 H route offers slightly lower primary allocation than W64.
The bounded1024 chunk reduced memory further but took49.871ms at1024;
bounded16384 raised allocation to423.074MiB without improving1024 time.
H recomputation at1024 took18.791ms and126.033MiB, versus saved H's19.264ms
and131.933MiB: this small difference needs independent replication.

Public Sphere optimizer validation is eager. No complete optimizer Graph
measurement is claimed. Actual complete-step allocated and reserved peaks
are measured after clearing independent-oracle scratch; GPU process total
is unmeasured. Separate phase/inference diagnostics and linear-only Graphs
are not added to or substituted for complete-step timing.

## Validation and unsuccessful attempts

Independent FP64 algebra checks all sites, Y, dX and every atom parameter,
with max-absolute and relative-L2 gates4e-4. Tests cover non-power-of-two axes,
chunk tails, rectangular/strided inputs, empty and single-site supports,
norm floors, exact capacity overflow, retained forwards, gradient branches,
live widths, and separate linear forward/backward Graph replay. Twenty-step
public optimizer comparisons use2e-6 absolute parameter and4e-4 gradient/
moment gates with exact step counters. The weight/support primary GPU suite
passed102 tests; the final126-test GPU suite, asymmetric widths, near-north/antipodal
centers, singleton/empty/floor profiles and batch64 also passed.

Failed jobs are retained:

- `48f14...`: correct math/20 updates, but a newly created binding attempted
  metadata refresh inside Graph capture. Bind outside capture.
- `09b6...`: retained autograd nodes from the snapshot test kept a default
  stream AccumulateGrad node alive. Use a clean graph and side-stream warmup.
- `c04d...`: singleton fixture omitted required amplitude/w_c constructor
  arguments. Its bitwise dX repeat check also incorrectly assumed atomic
  addition order was stable (observed difference7.45e-9). Correct the fixture;
  preserve the original19-atom case with the unchanged FP64 gate, bitwise dP,
  and a separate collision-free bitwise dX/dP snapshot check. No numerical
  oracle or optimizer tolerance was relaxed.
- `983d...`: a custom2e-6 floor fixture was rejected by the existing Torch
  backend. Restrict optimized supports to the existing revision1/1e-6
  contract, keep asymmetric-width/antipodal tests within that contract and
  add metadata rejection coverage. All numerical gates remain unchanged.
- `6c31...`: diagnostic inspection read a wrapper's saved tensor tuple instead
  of the underlying custom autograd node. Traverse the graph explicitly.

Primary source/result hashes, all negative measurements, correctness errors
and versions are in `20261008-sphere-polar-cuda-summary.json`. Raw driver,
source archives, verified result archives and logs are preserved under the
retained Sphere worktree's ignored `output/sphere-polar-blocked/`.

## Reproduction and integration

Use `plans-sphere-polar.json` and `cases/sphere-polar-*-sigma*.json` through
`python -m tools.kernel_dev prepare` then `check`, and
`python -m benchmarks.cuda.linear.run` as documented in the Linear README.
Sphere cases declare eager AdamW (`capturable:false`) and return `graph:null`.
The existing submission/DB Graph adapter does not support these records;
no DB observation or automatic selector promotion is claimed.

The2048 case uses bounded4096 as its executable baseline because the
original full-factor reference OOMs on L4; the original failure was not erased.
The final implementation checkpoint is `c14676f` (weight revision v2,
explicit validated patch tile32/64). Revision v1 used tile64 and spilled30 registers
in its weight VJP. V2 tile32 uses128 registers, zero spills and512 shared
bytes; assembly uses58 registers and zero spills. The inspected linear-only
Graph took4.985ms with117.582MiB allocated and226MiB reserved, versus
v1 tile64's6.742ms. This diagnostic excludes loss and optimizer.

An independent reverse-order existing-runner batch passed all four cases:
1024 width3/8,2048 width3/8. Every primary before/after full-shape FP64 gate,
20-update regression, retained forward and live-width linear Graph passed.
For width3 the v2 W32 route measured11.600ms /88.486MiB allocated at1024,
and30.562ms /308.008MiB at2048. W64 controls measured13.669ms and38.518ms.
Support64 H measured15.326ms/86.886MiB and50.505ms/302.008MiB. The original
1024 reference measured169.076ms, with the same4120.127MiB allocation.

For width8 bounded4096 measured19.460ms/131.933MiB at1024 and
102.716ms/283.854MiB at2048, versus bounded16384's108.700ms/863.395MiB.
The executable2048 bounded4096 baseline therefore gives a stronger memory
comparison than the initial16384 experiment. Initial Parameter/X/target
hashes match across every compared route and dense shares X/target hashes.
The separate CPU suite passed1349 tests with2378 GPU/DB skips; Ruff and
wheel/sdist builds passed. GitHub CPU checks are tracked on PR#80.

A second independent v2 W32/W64 run reverses their order. W32 measured
11.808ms at1024 and30.006ms at2048, with exactly the same allocated/reserved
peaks as the first run. W64 controls measured13.276ms and36.924ms. This
confirms the tile32 improvement across two independent GPU jobs, while
preserving the same initialization and full-shape oracle gates.

Supported patch tiles are limited to the GPU-validated32 and64 choices;
unused tile16 metadata is rejected. This final restriction changes no
measured execution code or plan. Earlier13-job source/result archives are
all verified and preserved, including failures and negative wide-W results.
The linear-Graph diagnostic retrieval was successful, but its remote file
cleanup request timed out; the pool stopped that owned runtime and the next
batch allocated a new session. Final runtime stop and PR integration status
are recorded after merge in the evidence manifest.
