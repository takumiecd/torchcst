# Fused Sphere preparation, 2026-10-09

Candidate source: `0aaadd80`, branch `kernel/sphere-fused-prepare`.
Research algorithm `research_cuda_sphere_polar_fused_weight`, revision `v1`.

The two explicit intrinsic S² charts, separable normalized Triweight profiles,
Polar amplitude/activity width law and task-width detach contract are unchanged.
Each forward snapshots source p, normalized live sites, embedded centres,
centre Jacobians, width precision, amplitude and its task Jacobian. The original
W32 support-patch assembly and shared-dW all-atom VJP remain unchanged. Supports
still scan all actual sites; count overflow takes the full traversal fallback.

The candidate replaces many Torch pointwise preparation launches/intermediate
arrays with two site-normalization CUDA launches and one grouped atom decoder,
then calls the existing two support pack launches. Scalar and radius tensors are
coerced to source dtype/device at execution; geometry revision 1 is explicit.
Algorithm instances retain no numerical tensor cache. Backward reads only the
forward snapshots, including after p/sites/radius/kernel scalar mutation.

Validation before performance:

- CPU: 1350 passed, 2397 skipped; Ruff/AST/declarations PASS; wheel/sdist PASS.
- L4 gate: 136 GPU tests, 0 skips, 111.37 seconds; existing runner64 PASS.
- GPU job: l4job-9e29ac82274e41fc9059e64883f671f9.
- Source SHA256: 139ae858306b2fef2629349dc40946c1e00e4d2474f8cc92f6b5e4d7800b57af.
- Result archive SHA256: f3b2265dd1c861396f953e95319260e2c21efe80fb6222ea7f40de4526cd0603.
- Hardware/runtime: NVIDIA L4, Torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0.
- Correctness: independent FP64 Y/dX/all 6 dP, cap overflow, tail atoms,
  empty/singleton/sub-floor profile, north/antipodal centres, asymmetric bounds,
  singular amplitude branch, mixed CPU FP64 scalar coercion, retained snapshots,
  linear Graph live widths and 20 public optimizer Parameter/moment/step checks.
- Existing near-sub-floor boundary gate is preserved: absolute FP64 error <=4e-4
  plus public FP32 reference comparison (atol4e-5/rtol4e-4). Ordinary full oracle
  uses both maximum absolute and relative L2 <=4e-4.

Small N64 smoke eager fused/old-W32 time ratio was 0.7377245754. This is a smoke
result, not the large-case adoption decision. Primary N1024/2048 B32 sigma3/8
complete eager steps and peak allocated/reserved memory passed the primary run; the qualifying N1024 sigma3 inverse run is pending. The sigma8
W256 baseline isolates preparation fusion; bounded H4096 is also a control and
must decide whether the candidate wins the actual wide-support workload.

Evidence: ignored output/sphere-fused-prepare/gate-v1, gate-v1-sha256.json,
cpu-tests.log, build.log, dist/, smoke_driver.py, measure_driver.py.

## Primary full-step results

B32, atom fraction 5%, FP32 IEEE, eager public CSTOptimizer, 3 warmups and
21 evolving updates. Independent full-size FP64 Y/dX/all-atom dP checks run
before and after those 24 updates. Timings below are complete steps; the phase
diagnostics are separately instrumented steps and are not additive estimates.

| N | Initial sigma | Old W32 ms | Fused W32 ms | Blocked H4096 ms | Dense ms | Fused allocated / reserved MiB |
|---|---|---|---|---|---|---|
|1024|3|11.368624|10.014453|18.331273|0.724338|88.486 / 110|
|2048|3|30.309803|29.507380|101.739289|1.137929|307.209 / 334|
|1024|8|128.806172|128.287489|18.995089|0.715790|245.287 / 256|
|2048|8|541.481160|538.980931|102.272923|1.103417|918.409 / 950|

Width3 uses support capacity64; width8 uses capacity256 for the isolated W
preparation comparison. The width8 W route loses decisively to bounded H4096.
N1024 sigma3 improves 11.91% with unchanged allocated peak. N2048 sigma3 improves
2.65% and allocated peak drops only 0.26%, below the predeclared 3% time or 5%
allocated-memory qualification. Only N1024 sigma3 qualifies for the independent
inverse order run. Sigma8 remains a negative result; fusion alone is not a
solution for the wider support workload. The public dispatcher is unchanged.

Primary job: `l4job-7270dbe192684a799e24e71478fc7d96`, driver319.96 seconds.
Source SHA256: `99274e22bd2d9afab43b459102978ea4460f2f3257e88f16aaac0af5f6beaea2`.
Result archive SHA256: `74c4a7d99f9a56784ea7557d2f4b0c819eb10836897a8c23bb3b830fb0490c00`.
Complete source/results/spec/receipt evidence and SHA256 manifest are preserved in
`output/sphere-fused-prepare/primary-v1` and `primary-v1-sha256.json`.

Reproduction: submit ignored `smoke_driver.py` with driver timeout900, then
`measure_driver.py` timeout1500 through the shared pool. The qualified followup
`inverse_driver.py` compares only N1024 sigma3 in fused/old-W/blocked order,
with dense and the same initial hash/24-update/full-oracle protocol.

The next hypothesis is exact conservative sorted-axis support scanning. Fusion
alone leaves the full O(A*N) profile scan, W patch work and eager optimizer
launches. That next candidate must keep all supports and per-side norm floors,
forward snapshots, live widths, exact overflow and full-step/memory gates.
