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

## Completed L4 results

Frozen source `3ae00284ac8fdcbe2d91f67b49dacd11c787f46f`; all1,483 source
file/driver hashes verified for each job, along with downloaded archive hashes.
Runtime files are identical to main `db818599`; this branch adds benchmark,
contract tests and notes only. CPU1343 passed/2267 skipped, Ruff and wheel/sdist
build passed. Each GPU job also passed the23 CPU regressions and20-step GPU
trajectory gate. Max GPU trajectory parameter error0; exp_avg3.725e-9,
exp_avg_sq5.457e-12 and dX1.630e-9. N64/all three contractions and N1024/both
contractions pass full FP64 checks before and after24 evolving training updates.
Maximum absolute Y/dX/all-dP errors across passing runs are approximately
3.462e-6/3.977e-6/3.445e-5; tolerances are unchanged.

Primary job `l4job-cb5b123add6a481c96001d84b05df3d7` completed in126.515s
of driver time. Independent reversed-order job
`l4job-47f3b309c4384467916ff947e021372a` completed all N1024 cases.
N1024 input/target/site/atom hashes and measured training peaks match exactly
between independent jobs. The companion summary retains case metrics and
source/result hashes; samples/logs remain in ignored `output/sphere-baseline/`.

| N / initial width | route | primary step ms | reverse step ms | allocated MiB | reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
|64 /3|public factored|10.364433|unmeasured|17.324|22|
|64 /3|factor-weight|9.982206|unmeasured|17.324|22|
|64 /3|public materialized|9.796063|unmeasured|24.081|42|
|64 /3|dense|0.717460|unmeasured|16.353|22|
|1024 /3|public factored|168.741845|169.015745|4120.127|4832|
|1024 /3|factor-weight|182.727245|182.111416|4120.127|4832|
|1024 /3|dense|0.733306|0.729977|32.751|42|
|1024 /8|public factored|169.668927|169.733729|4120.127|4832|
|1024 /8|factor-weight|184.200765|183.772500|4120.127|4832|
|1024 /8|dense|0.721511|0.711046|32.751|42|
|2048 /3|public factored, factor-weight|OOM in initial full check|not retried|incomplete|incomplete|
|2048 /8|public factored, factor-weight|OOM in initial full check|not retried|incomplete|incomplete|
|2048 /3|dense|1.136471|unmeasured|81.251|88|
|2048 /8|dense|1.129270|unmeasured|81.251|88|

At1024, factored is approximately7.2–7.9% faster than factor-weight with the
same measured peak. Plain inference is49.012–49.084ms factored versus
54.521–55.228ms factor-weight. Inference allocated/reserved peaks are
1463.061/1902MiB, including resident trained model/optimizer and inputs.
Separate instrumented primary factored rho3 phases: forward+loss49.129ms,
backward114.608ms, optimizer5.741ms. These are not summed into an alternative
step metric. Live width after24 steps is about2.99349 or7.94950 for both sides.
Width3/8 gives similar cost because this reference scans all site/atom pairs.

## Autograd memory diagnosis and2048 failure site

Separate instrumented diagnostic job `l4job-360aab2e5cf241368ffa6d81b7e7a984`
uses the exact same source, N1024/2048 sigma3 and public factored route. Its
new diagnostic scope has150s child/360s driver limits; it does not retry or
replace an OOM performance route. Raw traceback and saved-tensor shape histogram
are preserved. At1024 autograd requests two `[1024,52428,3]` geometry-offset
storages and many `[1024,52428]` profile intermediates. Distinct saved forward
storages total3,239,473,744 bytes (3.017GiB); this is saved storage accounting,
not a complete-step GPU peak. The `[32,52428]` H intermediate is only6,710,784
bytes (6.4MiB). Observed support is input mean28.266/output29.498 sites per
atom, zero empty atoms, against1024 scanned sites on each side.

At2048 the diagnostic fails already in **forward**, while computing the
output-side Sphere squared distance in `sphere.py:130`; no backward/optimizer
completion is claimed. CUDA has22.03GiB available total on the actual L4;
the process reports20.85GiB allocated and fails the next1.60GiB allocation.
The four original2048 route checks similarly fail at about20.88GiB allocated.
Those incomplete peaks include correctness work and are not training peaks.

Disposition: preserve the existing Sphere semantics and reference unchanged.
The measured reference is slow and retains excessive whole-site autograd
intermediates at this atom density. Bounded atom blocks, explicit VJP and exact
support-local execution are grounded next candidates; no speedup is claimed.
The factored H contraction is preferable to W assembly in the completed1024
cases. A single Sphere-chart profile product remains a separate mathematical
contract. These fixtures do not establish a Sphere-versus-Strip/Torus ranking.
This standalone diagnostic JSON is not the central runner/submission schema
and has not been inserted into PostgreSQL as standard benchmark observations.

Reproduce one route on CUDA:

```bash
python -m benchmarks.cuda.linear.sphere_baseline \
  --size 1024 --sigma 3 --route factored --output output/sphere-1024.json
```

The exact pool drivers, frozen source and receipts remain in `output/sphere-baseline/{primary,inverse,diagnostic}/` with original job IDs.

All owned L4 slots were verified stopped after the selected batch; no queued or
running jobs remain. Raw pool archives and receipts are retained in the research
worktree. Recovery files are `output/sphere-baseline/final-source.bundle` and
`output/sphere-baseline/final-manifest.json`; the managed worktree is retained
because the ignored raw evidence is needed. No runtime/default dispatch change
or Sphere profile-product API is part of this baseline.
