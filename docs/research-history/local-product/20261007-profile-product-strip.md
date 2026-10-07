# Polar profile product: whole-chart input Strip

## Contract and initial scope

The single Strip Chart spans all operator sites. Input tiles schedule computation;
they do not define independent atoms or independent normalization domains.
For atom a, whole input sites j and output sites i,

\[
W_a(i,j)=\mathrm{amp}_a\frac{u_a(i)v_a(j)}{D_a},\qquad
D_a=\max(\|u_a\|_2\|v_a\|_2,\epsilon).
\]

Input coordinates include the live pitch:
\(s_j=o_i+(j\bmod T)s+\lfloor j/T\rfloor P\).
The global norm, floor state, singleton certificates and center corrections
are computed once. Each small contraction receives this metadata restricted to
its input tile. Forward contributions and canonical atom VJPs sum across tiles.
Width task gradients remain detached; the existing Polar update evolves width.

Initial support: Euclidean Line axes with equal spacing, input Strip axis 1,
2..128 output sites, 16..1024 input sites, tile sizes 16/32/64/128, CUDA FP32,
batch 1..64, Polar triweight product. Chart parameter gradients are unsupported;
nontrainable live pitch is read on replay. This does not tile both axes.

The `tiled` recipe uses the small local contraction; `reuse` uses the tuned
ordered layout and launch settings from PR #59. The initial Python scheduler
visits each tile and retains its backward metadata. Correctness and complete
step time/allocated/reserved peaks determine the next optimization; no speedup
is assumed from tiling alone. These research plans are not public dispatch policy.

## Validation and evidence

The independent oracle constructs
full raw matrices per atom (chunked in atom count), normalizes over every site
pair, and checks Y, dX and all canonical gradients. Boundary/tail/floor cases,
source/scalar/pitch snapshots and captured Polar updates are separate checks.
Benchmark fixtures use output 64, input 256/512/1024, tile 64, pitch 68,
batch 32, approximately 5% atoms and initial width 3 or 8.


Actual NVIDIA L4 verification: 35 Strip tests (25 require CUDA), 66 Product
regressions and 83 existing core/owner tests passed. CPU suite: 983 passed,
1197 skipped; final CPU CI passed. wheel/sdist and Triton-free metadata import
passed. One initial setup failed because an installed `tests` package shadowed
a cross-test import; the independent fixture fixes it. No oracle was relaxed.

Complete-step measurements: NVIDIA L4, Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, FP32/IEEE, B32, output64, tile64/pitch68, fused AdamW and
production fused Polar update. Each execution has 21 synchronized Graph replay
samples; timings include forward, dX/all atom gradients and optimizer update.
Initial parameters, input and target hashes match across the CST routes.

| input / atoms | rho | Torch us | tiled us | reuse us | dense us |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 / 819 | 3 | 313.94 | 569.80 | 201.91 | 45.31 |
| 256 / 819 | 8 | 313.93 | 570.07 | 201.50 | 45.32 |
| 512 / 1638 | 3 | 414.34 | 2075.64 | 498.91 | 49.66 |
| 512 / 1638 | 8 | 415.37 | 2076.85 | 493.52 | 50.00 |
| 1024 / 3276 | 3 | 1975.97 | 7848.76 | 1739.16 | 54.36 |
| 1024 / 3276 | 8 | 1980.81 | 7849.44 | 1734.10 | 54.43 |

Independent reverse-plan-order rho3 executions: 256 Torch/reuse/dense
309.85/200.60/45.00 us; 1024 1976.33/1739.85/54.49 us. Thus rho3 has two
executions at 256/1024 and one at 512; rho8 has one execution per size.
These are descriptive observations, not a significance test. No unfavorable
case is omitted: reuse loses to Torch at512 and remains far from dense.

| input | reuse allocated bytes | reuse reserved bytes | Torch allocated bytes | dense allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| 256 | 727552 | 6291456 | 43066368 | 34425344 |
| 512 | 2264576 | 8388608 | 67962880 | 34753024 |
| 1024 | 7856128 | 18874368 | 161384960 | 35408384 |

Peaks include capture/replay and warmed model, gradients and optimizer state.
They are allocator-accounted memory, not GPU process usage. At256, Torch and
dense have a common additional 34078720-byte allocation before capture versus
the custom routes. Its internal attribution was not instrumented; the memory
gap must not be described as solely model-tensor storage. All route peaks,
including reserved bytes, are in the accompanying machine summary.

Across the eight benchmark executions, maximum absolute oracle discrepancies
are Y6.08e-6, dX2.09e-6, dp3.78e-6; the shared-gradient Polar update matches
exactly. Tests also cover complete captured update trajectories, live pitch,
source snapshots, floor behavior, boundaries and partial tails.

## Existing Strip/Torus scheduling and next direction

The existing `strip_torus/fused` runtime prepares/routs/packs atoms once and
launches the whole tile grid from one autograd operation; its forward and dX
kernels loop over reduction tiles on device. The prior faster bounded prototype
builds conservative per-tile atom lists, creates window-local W and uses GEMM
with a bounded/cache policy. See [the shortlist](../cuda-linear/cuda-kernel-shortlist.ja.md)
and [the historical L4 profile](../cuda-linear/notes/l4-current-bottleneck-20260929.ja.md).
Its archived code remains at commit190a1bb3fa6259fde493a30d901d5cab7c2ebb82
(`prototypes/block_streamed_backward.py`, `block_materialize_listed.py`,
`block_tile_atom_lists.py`). This inspection did not remeasure the old route.

The old runtime uses Torus geometry, Direct amplitude/width and raw radial
profiles; current experiments use Euclidean Polar product with whole-operator
L2 normalization. Old timing is therefore not a matched speedup comparison.
Its scheduling and conservative support-list design are nevertheless relevant.
Retain the globally normalized Strip implementation as the correctness baseline;
next compare once-per-forward support preparation plus a combined tile grid,
and bounded local-W/GEMM scheduling. Preserve global center corrections,
canonical Parameter/moment identities and per-forward snapshots. The current
per-tile all-atom layout and separate autograd calls are concrete repeated work;
their measured contribution still needs profiling. Do not claim a bottleneck
solely from inspecting launch counts.

## Evidence and reproduction

[Machine summary](20261007-profile-product-strip-summary.json) records each
job, committed source, source/result archive SHA256, driver/result/snapshot
hashes and metrics. Every frozen tracked file was checked against `git archive`
of the recorded commit. Benchmark source is
`ec87b1bdfcc10324ec738440ec81ddb8c923230f`; initial GPU regression source is
`e203b088bd45ced730415a1a427d31bbcf19515e` (later source changes are metadata
support checks, tests and documentation). All eight artifacts pass projection
revision5 and the public submission consistency check. No central DB write or
public dispatcher promotion was performed.

Raw evidence stays in ignored
`benchmarks/cuda/linear/evidence/profile-product-strip-20261007/` and
`~/.local/state/colab-l4-pool/jobs/<job-id>/` (including the setup failure).
All jobs completed and pool-owned runtimes were verified stopped.

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-strip.json \
  --case benchmarks/cuda/linear/cases/profile-product-strip-256-rho3.json \
  --candidate tiled --candidate reuse --output output/strip-comparison
python -m tools.kernel_dev check \
  --plans output/strip-comparison/plans.json \
  --case output/strip-comparison/case.json
python -m tools.kernel_dev test --suite profile-product-strip
python -m benchmarks.cuda.linear.run \
  --plans output/strip-comparison/plans.json \
  --case output/strip-comparison/case.json --polar-update fused \
  --source-commit COMMIT --output output/strip-comparison/result.json
python -m benchmarks.submissions check output/strip-comparison/result.json
```
