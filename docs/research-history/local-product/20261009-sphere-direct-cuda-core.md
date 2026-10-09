# Sphere CUDA Core direct contraction study

The user prioritizes CUDA Core execution for the existing Sphere operator.
This study keeps the two intrinsic S² Explicit charts, chord distance,
Polar/Triweight profiles, separate complete-chart L2 floors, live widths,
width stop-gradient task VJP, and all six atom parameter gradients.
It changes execution only. Public dispatch and API are separate decisions.
The operator/chart contract is described in the
[Sphere kernel profile study](20261009-sphere-kernel-profile.md).

## Fixed initial candidates

All new routes reuse the existing fused preparation and lossless int16
support IDs with int32 address widening. CAP64 and site tile 16 are fixed.
The four routes cross atom groups 1/4 with separate/merged output VJP.
They retain H for backward, keep G within each contraction CTA, and allocate
neither square W/dW nor complete sites-by-atoms U/V arrays. Bounded packed
profile values and support IDs remain saved for backward reuse; this study
does not claim their calculation or storage has disappeared.

The old support64 direct route is a mechanism control. It also differs in
preparation, site-ID storage, and site tile size, so comparison against that
route alone does not isolate grouping or VJP fusion. The four new routes
have identical preparation and isolate those two changes. The accepted
compact G8/T16/int16 W route is the performance and memory adoption reference.

## Single-pass output gradient algebra

For one atom, write raw output profile r_j, n=||r||₂,
t=max(n,floor), and u_j=r_j/t. Let h_b be the saved input contraction,
g_b=sum_j dY_bj u_j, and c_j=amp sum_b h_b dY_bj. The raw derivative with
respect to the ambient Sphere centre q is

\[
D_j = 6\,\mathrm{precision}\,(1-\mathrm{precision}\|s_j-q\|^2)_+^2(s_j-q).
\]

The first output traversal accumulates G and three-component moments

\[
M=\sum_j c_j D_j/t,\qquad L=\sum_j u_j D_j/t.
\]

After G is complete, let p=amp sum_b h_b g_b. The centre gradient is

\[
\nabla_q\mathcal L=M-\mathbf1_{n\ge\mathrm{floor}}\,pL.
\]

This is the same complete-profile normalization VJP as the old two-pass
calculation. Below the floor, only M remains. The final ambient gradient is
pulled back through the existing intrinsic centre Jacobian. Equality is
algebraic; FP32 rounding and atomic reduction order still require independent
numerical validation. No speed or memory improvement is established by this
formula alone.

## Predeclared measurement contract

N1024/N2048, B32, floor(0.05*N²) atoms, seed41, initial sigma3, FP32 IEEE,
TF32 disabled, MSE and existing fused AdamW/research Sphere update Graph.
Each isolated worker checks full-site/all-atom FP64 Y, dX and all-dP both
initially and after 24 actual updates with maxabs AND relative-L2 <=4e-4.
Two warmup updates, one initial Graph replay and 21 timed replays total24.
Peak allocated/reserved include capture/replay. Separate phase diagnostics
are never summed into or subtracted from the uninstrumented complete step.

The fixed initial cohort contains compact W, old direct, all four new
ablations and dense at both sizes, without filtering. All correctness workers
must pass before timing. A candidate qualifies against compact W with either
strictly more than 3% speed improvement, or at least5% allocated reduction
and no more than 3% time regression. Adoption requires independent reverse
confirmation at both shapes. Improvements against old direct alone are
mechanism evidence. Failed or negative outcomes remain preserved.

The ignored protocol is output/sphere-direct-cuda-core/protocol.json,
SHA256 d598c11d2c0bf4b274ce73da53754dd327db68e096a479a9088b8a57d5d12c22.
Budgets are900s outer,875s driver,350s child. Runtime source, driver, plan and
result hashes for completed validation and performance appear below.

## Initial validation failures

Source b0a8edb was preserved with job
l4job-dbcf30dde1ef49d29a65bfe2f1105c8e. Its single pytest child exceeded350s
after three retained-forward assertions failed. No cohort worker ran and this
job supplies no accepted performance result. The fixed900/875/350s budgets
and numerical gates remain unchanged.

The separate diagnostic l4job-aeb622904ddf4456a1568b64f98467b1 reproduced all
three failures in dX only (192 elements, maximum absolute differences
1.4901161193847656e-8,7.450580596923828e-9,7.450580596923828e-9).
The test incorrectly demanded bitwise repeatability of atomic dX reductions.
Commit c3557d8 checks repeated dX with the existing maxabs AND relative-L2
gate and retains bitwise dP equality and independent FP64 checks before and
after mutation. The runtime is unchanged by this correction.

The first failed gate also returned partial G1 compiler diagnostics:
forward 39 registers, backward 80, zero spills and 512 bytes shared memory,
for both separate and merged VJP. These are diagnostics from an incomplete
job, not evidence that all variants passed or that a candidate is faster.
The replacement gate partitions the same 159 tests without omission or retry.
All partitions and all 14 correctness workers passed before timing.

## Completed fixed cohort: negative adoption result

The partitioned gate l4job-d5540038f89747358b2edd4ab07b8153 passed 159 tests
with zero skips and all 14 full correctness workers in 763.31s. Its six
partitions contained 98/1/1/1/1/57 distinct tests. The independent source,
JUnit and raw-artifact audit passed. Maximum absolute oracle error was
5.6681e-5, within the unchanged4e-4 gate.

Primary l4job-e1735b731bae4f05964b6e19b343088d passed 30 protocol tests
and all 14 workers in 374.92s. Both jobs used frozen source
5b8e1d8d7353ec24b3a394d8e3ddc50b7b36e372 and partitioned-v2 driver
SHA256 1b6ddd34c52f0e71c63b3c092108bcd3d49cbd390c318e915243d4b38ee5b6a8.
Hardware was NVIDIA L4; Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0 and
Triton 3.6.0. Complete Graph step medians and capture/replay peak allocated
memory were:

| Route | N1024 ms | N1024 MiB | N2048 ms | N2048 MiB |
| --- | ---: | ---: | ---: | ---: |
| compact W | 4.164 | 92.862 | 22.562 | 271.585 |
| old direct | 8.928 | 87.662 | 43.700 | 302.259 |
| G1 separate | 8.024 | 74.862 | 40.080 | 250.259 |
| G4 separate | 6.414 | 74.862 | 33.289 | 250.259 |
| G1 merged | 7.487 | 74.862 | 37.035 | 250.259 |
| G4 merged | 6.051 | 74.862 | 31.697 | 250.259 |
| dense control | 0.0855 | 32.877 | 0.5937 | 81.502 |

All four new routes reduced allocated memory versus compact W, but failed
the predeclared time constraint. G4 merged was45.29% slower at 1024 and
40.49% slower at 2048, with 19.38% and 7.85% allocated reductions. The frozen
assessor returned an empty eligible selection; no reverse adoption run or
public-dispatch change followed. Improvement over old direct is mechanism
evidence only, and CST still has no measured memory advantage over dense here.

Separate instrumented diagnostics at 2048 measured compact W forward/loss
14.569ms and backward7.220ms; G4 merged measured18.830ms and 12.748ms.
These separate30-update runs are not added to or subtracted from primary
timings. Neither CUDA Core grouping nor output-VJP fusion removed atomic
scatter. Atomic dominance is a hypothesis, not an isolated measurement.

Matched2048 compiler diagnostics found G1 forward 39/backward 80 registers
with zero spills; G4 forward 71/backward 168 registers, with 2 backward spills
for separate VJP and 0 for merged. No MMA/TF32 instructions were present.
These explain resource differences but do not establish a timing cause.

Raw jobs, frozen-v2 driver/protocol/assessor and their source/result archives
are preserved in the primary checkout under
output/sphere-direct-cuda-core-20261009/. The retained
`kernel/sphere-direct-cuda-core` branch and commit
`5b8e1d8d7353ec24b3a394d8e3ddc50b7b36e372` preserve the implementation.
The candidate code, tests and catalogs are not in `main`; this integration
contains research notes only. No candidate is selected as a public default.
All owned L4
sessions were verified stopped after the completed batch. The next isolated
study removes Phi storage at allocation time and recomputes profiles from
saved support/geometry/norm; its completed results are recorded in the
[profile recomputation study](20261010-sphere-direct-recompute.md).
Those measurements are distinct from the 107374080-byte capacity estimate at 2048.

Final recovery is preserved under
`output/sphere-cuda-core-recovery-20261010/` in the primary checkout.
`manifest.json` records exact restoration of all three named research refs,
presence of every frozen measured commit, and `git fsck` exit code 0.
The history bundle SHA256 is
`6845e7966ab5509e1c9f48d75e54dedc49a338cc02d0559828e48f686bc39b71`.
`evidence-manifest.json` records 608 raw/tooling files restored byte-for-byte;
the evidence archive SHA256 is
`a7d8970176bbbf32d9f041177c8db5a4779f05a843c08e43caee222431e1b6ac`.
`gpu-stopped.json` records every owned pool slot stopped. No branch or worktree
was deleted. Candidate code remains on the local named research refs; this
integration contains notes only. Preservation follows the
[research operating rules](../../research-operations.ja.md).
