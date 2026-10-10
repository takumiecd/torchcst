# Regular Product: bounded CSR matrix-free protocol

This is a new execution decomposition after the rejected metadata-recompute and
compact-cache W/dW routes. Equal positive Line spacing is already part of the
matrix control; the experiment isolates removal of W/dW, global H/G and sorting.
Public dispatch is unchanged. Research algorithm:
`research_regular_product_matrix_free/v1`.

For each atom, let `v` and `u` be the input/output Triweight profiles and let
`D = max(||u|| ||v||, floor)`. The prepared factors retain the existing whole-product
floor convention, singleton flags and live activity-dependent inverse variance.
Then `H_ab = sum_j X_bj v_aj / s_v`, `Y_bi = sum_a amp_a H_ab u_ai / s_u`, with
`s_u s_v = D`. Backward computes `G_ab = sum_i dY_bi u_ai / s_u`, input/source VJP
and canonical Polar cotangents with detached task width. No W or dW is allocated.

Preparation retains the exact 13-field forward snapshot (52 bytes/atom). H/G,
owner counts/offsets/cursors and ID lists are reused across bounded atom chunks.
CSR membership uses prepared positive support intervals and 16-site owners.
Normal atoms enter at most four owner lists; broader atoms enter a disjoint
complete overflow list inspected by every owner. The capacity limit never
truncates a support or atom. This overflow path is a correctness fallback whose
cost is included in the full step. CSR order is temporary; Parameter and Adam
moments retain canonical atom order. Backward rebuilds input owner lists from the
saved forward factors; it never uses updated widths or centers.

## Frozen comparison

Two candidates only: atom chunks 65,536 and 262,144. Both use atom group8,
preparation group16/sites16, owner block32, four owner splits and capacity4.
Native Triton and Torch large-matrix controls use group8/patch8 and the same
support preparation. Dense is included as a separate parameterization comparison.
B32, seed41, K=floor(0.05 N²), initial rho3, live width bounds1..16,
FP32 IEEE/TF32 disabled, fused capturable AdamW lr1e-4/wd0.01 and public Polar
update. Runner warmup5/rounds21 and isolated phase diagnostics match the preceding
Product measurements. Report capture/replay allocated and reserved peaks separately.

Initial and after24 live Graph updates: independent full-site FP64 checks of Y,
dX and all four canonical cotangents for every atom. Maxabs AND relative-L2 <=4e-4.
Boundary tests include non-square axes, chunk tails, strides, empty/singleton,
floor and precision fallbacks, wide overflow, retained forwards and requested grads.
No truncation, oracle sampling or relaxed tolerance is allowed.

N2048 first: GPU boundary gate plus full comparison/updated oracle, driver deadline
1,200 seconds. Only after all correctness gates pass, N8192 full comparison and
updated oracle, deadline1,800 seconds. These deadlines include driver setup;
pool transfer/verification has its own limits. An unchanged failed freeze is not
retried. A source/fixture repair must preserve the failure and state its reason.

Speed qualification: >3% faster with no allocated peak increase. Memory
qualification: >=5% lower allocated peak with <=3% time regression. Both valid CST
controls must be reported. A qualifying result receives one independent reverse
plan-order run with the same cases, controls, oracle and deadline. No promotion
from isolated phase timing or from tensor budgets; unchanged source comparisons
use explicit full Git SHA. Record raw job/source hashes and disposition before PR.
