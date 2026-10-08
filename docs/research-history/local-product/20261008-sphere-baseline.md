# Existing two-Sphere Polar baseline

The owner requested ordinary Sphere speed and memory following the Torus study,
then asked to inspect the existing implementation first. This experiment keeps
the existing `polar_activity` declaration and public `CSTOptimizer` unchanged.
It does not introduce a Sphere profile-product declaration or a single-Sphere
operator chart. The existing Polar kernel requires two charts, with separate
chart-site L2 normalization and a 1e-6 floor on each profile. This differs from
the Torus profile product's single whole-operator floor and coupled distances.

The fixture uses two S2 charts with uniformly sampled ambient observation
sites, intrinsic centers and radius sqrt(N/(4*pi)). Sphere surface area is N,
so mean site density is one; the radius is an explicit fixture choice, not a
public default. Each chart has N sites. B32, 5% atoms, seed41, FP32 IEEE/no
TF32, widths initially3/8 under live Polar activity, AdamW lr1e-4/decay0.01,
and dX are retained. Input centers retain public uniform sampling and output
centers public balanced sampling. All routes receive identical sites, atom
parameters, inputs and targets, recorded by SHA256.

Three contractions are identified separately:

- `factored`: actual public Torch factored forward, retaining its full factors
  and batch-by-atom intermediate, plus public eager CSTOptimizer.
- `factor-weight`: benchmark-only W=output_factor @ input_factor.T, then XW.T,
  using the same existing factors and public eager CSTOptimizer. This avoids
  atom-by-output-by-input materialization without changing the kernel.
- `materialized`: actual public materialized operator, tested/measured at64.
  Its atom contribution tensor alone is about204.8GiB at1024 (52,428 atoms)
  and3.2TiB at2048; these are tensor lower bounds, not measured GPU peaks.
  It is not submitted at large sizes because this single allocation exceeds L4.

A normal dense layer is a speed/memory reference and a different model.
Measurements are eager synchronized wall-clock complete steps, with 3 warmups
and21 evolving samples. The actual public optimizer rejects CUDA capture,
so these cannot be compared directly with Torus Graph medians. Allocated and
reserved training peaks include optimizer initialization/warmup and complete
steps; oracle scratch is cleared before measuring. Resident trained state is
retained for inference, and that scope is recorded separately. Total process
GPU memory is not measured. Instrumented phase timings are diagnostic only;
their medians are not added to replace the complete-step timing.

Independent FP64 physical embedded-distance/Polar task-map oracle checks every
site and atom for Y,dX,all dP initially and after24 training updates, with both
maximum absolute and relative L2 tolerance4e-4. Twenty optimizer steps compare
factored/W contractions at N32, including all parameters, dX, moments and step
counts (parameter max2e-6, other max4e-4). No tolerance relaxation, support
sampling or dropped unfavorable case is permitted.

First batch: N64 sigma3 (all three CST routes and dense); N1024/2048 sigma3/8
(factored, factor-weight and dense). Each route runs in a fresh process, child
limit180s, batch driver limit1800s plus100s setup/teardown. A timeout/error is
recorded explicitly and is not retried with a higher budget. OOM is a measured
failure, not a correctness/performance pass. Reverse-order independent batch
is planned only for completed passing N1024 cases, with identical limits.
No universal Sphere/Torus speed winner can be inferred across different models.

CPU preflight: 23 tests passed (new physical oracle and trajectory checks plus
existing Sphere geometry regression). GPU and performance results pending.
