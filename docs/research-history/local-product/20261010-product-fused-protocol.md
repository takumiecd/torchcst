# Regular Product: fused source backward

This experiment follows the bounded CSR matrix-free route. At N8192 that route
reduces allocated peak substantially, but its complete step fails the joint gate.
The separate backward phase takes about28ms. The new route changes one mechanism:
compute H and its input-center derivative Hci together in one input pass inside
the source VJP. Forward, output G/Gco, normalization, support preparation, owner
membership, overflow, canonical Polar state and scratch allocation stay identical.
Research algorithm: `research_regular_product_matrix_free_fused/v1`.

For normalized input/output factors vn and un, retain the exact forward snapshot.
Compute H=sum_j X_j vn_j and Hci=sum_j X_j d(vn_j)/dci from the same X load.
Then dAmp=sum_b H_b G_b, dCo=amp sum_b H_b Gco_b and
dCi=amp sum_b Hci_b G_b. Singleton input factors retain H and have Hci=0.
Whole-product floor D=max(||u|| ||v||,floor), detached live task widths and all
four canonical Polar cotangents keep their existing mathematical contract.

## Frozen comparison

One new candidate: atom chunk262144, group8, preparation group16/sites16,
owner block32/splits4/capacity4. Compare the unfused CSR256 control and native
Triton/Torch group8/patch8 matrix controls. Dense remains a separate model
comparison. No new support/cache/owner policy or precision setting is tuned here.
Use `plans-regular-product-fused.json` and the N2048/N8192 rho3 case files.
B32, seed41, K=floor(.05 N²), width bounds1..16, FP32 IEEE/TF32 off,
fused capturable AdamW lr1e-4/wd.01, public Polar update, warmup5/rounds21.
Primary Graph timing includes forward, backward and optimizer. Separate phase
diagnostics continue evolving widths and are not summed into the primary timing.
Report capture/replay allocated and reserved peaks; process memory is unmeasured.

Full-site FP64 Y/dX/all-atom/all-four dP checks require maxabs AND relative-L2
<=4e-4, initially and after24 real updates. The supplemental driver performs one
eager update and23 Graph replays, then asserts Adam's actual step counter is24.
Capture records the step without executing an update. Earlier driver artifacts
labeled24 performed23; retain those raw artifacts and use separate corrected
oracle-only jobs, without rerunning their performance. Main benchmark itself
performs52 real updates (5 eager warmup,21 timed eager,5 side warmup,21 replays).

Boundary GPU gate includes original matrix-free tests, fused variants and existing
matrix/global/dispatch regressions. Fused tests retain empty/singleton/floor,
large-origin fallback, overflow/tails/strides, retained forward snapshots,
requested gradients and20 replay updates; B1/B64 both groups/preparations added.
No sampled oracle, relaxed tolerance, truncated overflow or dropped zero amps.

N2048 gate+comparison+oracle deadline1200s. Only after correctness passes,
N8192 comparison+oracle deadline1800s. These include driver setup; transfers have
separate pool limits. A source/fixture repair preserves the failed freeze and
reason. No unchanged failed comparison is retried.
Qualification against BOTH native and Torch CST controls: >3% faster with no
allocated increase OR >=5% lower allocated with <=3% time regression. Report the
unfused control separately to isolate the fusion. A qualifying result receives
one reverse-order independent confirmation with unchanged protocol/deadlines.
Only complete-step evidence determines promotion. Record full source SHA,
verified raw hashes and disposition before any PR integration.
