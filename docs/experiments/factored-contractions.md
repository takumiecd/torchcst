# Compact contractions instead of visible derivative tensors

## Implementation

Opt in with `ImplicitAdamConfig(factored_geometry=True,
quartic=DeviceRay(corrections=1), device_execution=True)`. Kernels must advertise
exact input/output factorization. The default visible path remains unchanged.

For W(p)=sum_k v_k(p_k) u_k(p_k)^T, differentiate only u and v. Their values,
Jacobians, and local Hessians occupy O(K*(I+O)*(1+P+P^2)) storage. Directional
Taylor values use the product rule and matrix products; pullbacks use the
corresponding adjoint contractions. No visible J[K,I*O,P] or H[K,I*O,P,P] is
built on this path.

At 1024x1024, K64/P4, factor values and derivatives total 10.5 MiB, versus
5 GiB for visible J/H alone. The factor-local Hessians are still present;
this is not a claim that all Hessians or all I-by-O tensors disappear.
The visible metric and represented directional displacements still use O(I*O)
storage. No giant K-by-I-by-O-by-P-by-P inputs or layout clones are needed.

Exact operations covered:

* Jd and H[x,d] from factor directional derivatives, and their adjoints for
  the original quartic value and gradient.
* The diagonal preconditioner from separable weighted factor inner products,
  including the metric epsilon term.
* Accepted-frame Gram construction from inner products of four rank-one
  product-rule terms per column, retaining all cross-atom interactions.
* Previous-frame moment transport and its affine pullback.
* Backward observations: differentiate the scalar
  sum((X u_k) * (output_gradient v_k)), avoiding even the intermediate atom
  weight matrix. Capture and replay these AD contractions as well.

The factor AD transformations execute during graph warmup/capture. Small
explicit factor tensors feed compiled contractions and CUDA Graph replay.
CPU synchronization is permitted at cold setup and explicit reporting
boundaries, not inside warmed measured updates. Buffer changes invalidate
capture caches as on the existing device path.

The moment-compression solve is still the existing float64-accumulated,
twelve-sweep Jacobi pseudoinverse. This work does not change its rank cutoff,
validation, or mathematical role. It remains a likely fixed-cost bottleneck.

## Validation protocol

Tests compare directional derivatives, contracted Hessians, pullbacks, frame
Grams, metric diagonals, scalar ray coefficients, objective/gradient values,
and selected ray steps against the full visible derivative oracle. Both
float32/float64 and amplitude/kernel variants are included. CUDA graph tests
exercise changing frame inputs and the synchronization-error guard. A complete
CPU optimizer test forbids calls to the visible derivative-cache builder.

Scaling uses the same A100 MIG 3g.40gb, batch128, K64/P4, float32,
`highest` matmul precision, synthetic resident batches and three timing blocks
as [the preceding sweep](scaling.md). New runs use 32 steps/block at all widths,
including 1024, and do not override the derivative-cache limit. Compilation
and eight warmup steps are excluded. Peak tensor allocation is reset after
warmup; it excludes the compilation peak and includes retained CUDA graphs.

MNIST uses the previous 8192/2000 split, 128 updates, batch128 and seeds
17/29/43, with the same restored-after-warmup protocol as the previous Ray 1
measurements. Contraction order can perturb ill-conditioned moment compression
and alter the training trajectory; oracle agreement does not establish
identical learning accuracy.

## Results

| Width | Previous Ray ms | Compact Ray ms | Previous peak MiB | Compact peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 128x128 | 45.42 | 30.25 | 347.3 | 71.8 |
| 256x256 | 77.39 | 31.00 | 1308.2 | 81.8 |
| 512x512 | 228.20 | 32.71 | 5150.7 | 104.7 |
| 1024x1024 | Failed | 37.00 | No steady-state result | 152.3 |

512x512 is about 7.0 times faster with 49.2 times less peak tensor memory.
The old 1024 run failed during warmup; its 20.3 GiB allocated / 36.6 GiB
reserved peaks are not a successful steady-state baseline. The compact 1024
run succeeds at 152.3 MiB allocated, 594 MiB reserved, and 37.00 ms/step.
It does not need an expanded derivative-cache limit. Cold setup including
eight warmup steps is about 10–11 seconds per shape in this run; it is not
included in the timing table.

Relative to the previous Dense measurements, compact Ray is approximately
56x/56x/62x/69x slower at widths 128/256/512/1024. The earlier severe scaling
trend is substantially reduced over these dimensions, but Dense-like training
speed is not achieved. These comparisons retain fixed K=64; capacity and
accuracy are not matched to Dense.

MNIST (128 steps):

| Seed | Previous visible Ray accuracy | Compact Ray accuracy | Compact Ray seconds |
| --- | ---: | ---: | ---: |
| 17 | 77.45% | 75.55% | 4.093 |
| 29 | 74.60% | 75.75% | 4.049 |
| 43 | 76.50% | 76.80% | 4.050 |
| Mean | 76.18% | 76.03% | 4.064 |

The mean accuracy difference is -0.15 percentage points, with a -1.90 point
change on seed17. This small sample does not establish equivalence or an 80%
accuracy guarantee. The compact path remains opt-in. The previous visible
Ray mean time was 5.647 seconds; the historical ordinary CST+Adam mean was
0.926 seconds, so compact Ray is about 4.4 times that baseline.

Warmed MNIST phase measurements: transport 2.27 ms, ray solve 0.51 ms, Gram
construction 1.61 ms, Gram solve 24.93 ms. Phases are inclusive stream times,
may include queue idle, and should not be summed as independent CPU/GPU costs.
The remaining dominant numerical operation is moment compression.

The optimizer-update profiler audit found no scalar host reads, host
synchronization, or device-to-host events inside the measured update. All
complete training loops also passed the CUDA synchronization-error guard.
Local tests: 198 passed, 22 skipped. Targeted A100 suite: 17 passed. The later
strengthened no-atom-materialization test was verified locally. Ruff and diff
checks pass. All remote runs completed successfully.

Raw results: `output/factored_a100/output/`. Implementation commits:
`6bb345f` (compact geometry and ray contractions), `7671260` (backward
observations). The intermediate geometry-only experiment is retained under
`factored/`; it still materialized backward atom weights and is not the final
implementation measured in `final_factor/`.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.scaling_benchmark \
  --method cst_ray1 --inputs 1024 --outputs 1024 --factored-geometry \
  --output output/final_factor/1024.json
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --factored-geometry --methods ray1 --seeds 17 29 43 \
  --data /path/to/MNIST/raw --output output/final_factor_learning
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --factored-geometry --audit-only \
  --data /path/to/MNIST/raw --output output/final_factor_audit
```
