# Sphere CUDA Core direct contraction study

The user prioritizes CUDA Core execution for the existing Sphere operator.
This study keeps the two intrinsic S² Explicit charts, chord distance,
Polar/Triweight profiles, separate complete-chart L2 floors, live widths,
width stop-gradient task VJP, and all six atom parameter gradients.
It changes execution only. Public dispatch and API are separate decisions.

## Fixed initial candidates

All new routes reuse the existing fused preparation and lossless int16
support IDs with int32 address widening. CAP64 and site tile16 are fixed.
The four routes cross atom groups1/4 with separate/merged output VJP.
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
initially and after24 actual updates with maxabs AND relative-L2 <=4e-4.
Two warmup updates, one initial Graph replay and21 timed replays total24.
Peak allocated/reserved include capture/replay. Separate phase diagnostics
are never summed into or subtracted from the uninstrumented complete step.

The fixed initial cohort contains compact W, old direct, all four new
ablations and dense at both sizes, without filtering. All correctness workers
must pass before timing. A candidate qualifies against compact W with either
strictly more than3% speed improvement, or at least5% allocated reduction
and no more than3% time regression. Adoption requires independent reverse
confirmation at both shapes. Improvements against old direct alone are
mechanism evidence. Failed or negative outcomes remain preserved.

The ignored protocol is output/sphere-direct-cuda-core/protocol.json,
SHA256 d598c11d2c0bf4b274ce73da53754dd327db68e096a479a9088b8a57d5d12c22.
Budgets are900s outer,875s driver,350s child. Runtime source, driver, plan and
result hashes will be recorded with completed jobs. Implementation and GPU
validation are pending; this initial note reports no performance result.
