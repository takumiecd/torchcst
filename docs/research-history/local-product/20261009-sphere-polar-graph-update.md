# Existing Sphere Polar complete CUDA Graph updates

The ordinary model remains two intrinsic S² charts, per-side chart-site
Triweight L2 normalization with a 1e-6 floor, and the live Polar activity/width
law. No single-chart Sphere profile-product definition is introduced.

The research CUDA `atom_update` plan uses the common AtomUpdateBinding,
AtomUpdateInputs and Dispatcher. AdamW retains its rounded proposal, moments
and step counter. The Polar update projects task displacement tangentially,
updates the radial activity clock and applies its existing exact radial decay.
For each intrinsic centre pair c, retraction is the public operation

    c' = c + (proposal_c - c)
    limit = radius * (pi - chart_margin)
    c_new = c' * min(limit / max(norm(c'), tiny), 1)

Radius and margin are independent live device scalars for both charts.
Intrinsic gradient projection and vector-state transport are identity, as in
the public Sphere geometry. Metadata rejects other geometries, representations,
shapes, precision, profile/floor contracts and Polar activity modes. Benchmark
preflight additionally rejects malformed radius/margin tensor metadata before
any base-optimizer proposal.

The public CSTOptimizer remains eager, with its existing synchronous finite
checks and geometry validation. `benchmarks.cuda.linear.sphere_graph` is an
explicit research base-AdamW proposal plus declared update Plan. Its finite
initial-state and trajectory gates do not reproduce public synchronous failure
semantics during replay. Thus public eager versus research eager compares both
the updater and that wrapper boundary; research Graph versus research eager
isolates capture. A public-eager/Graph ratio is not a kernel-only speedup.

Complete steps include Y, loss, dX, all six atom gradients, AdamW moments/clock
and Sphere retraction with evolving activity width. Graph timing uses two
warmup updates, one capture update and 21 replays; eager timing uses three
warmup updates and 21 measured updates. Peaks include capture/replay, distinguish
allocated and reserved, and exclude FP64 oracle scratch. Dense peaks retain only
the dense model/optimizer and shared X/target, not a Sphere model. Registered
Linear Plan JSON can be supplied for combined preparation experiments.

## Verified correctness checkpoint

- CUDA source checkpoint: `3632f8c4e1f4ac105acc57d53bc3b932e12f062c`.
- NVIDIA L4 gate: 39 passed, zero skips, 10.33 seconds. Both activity modes,
  live asymmetric radii, north/cap centres, zero/under/over-radius Polar points,
  20 captured proposal/state updates and 20 complete training steps.
- Complete-step trajectory uses independent Torch factored linear execution
  and public CSTOptimizer. Parameter absolute tolerance is 2e-6; other state/
  trajectory absolute tolerance 4e-4; step counters are exact. The large runner
  separately gates full-shape Y/dX/all dP against the FP64 oracle at 4e-4
  absolute and relative L2, before and after training.
- Job: `l4job-fbc49171eebc4a0a97d33278ae5f7af4`.
- Source archive SHA256:
  `6f1043203bebab6b0ecb605cdf0127397350221ab4bf26afea3caecae1596005`.
- Result archive SHA256:
  `4e48e3ef4529077932f109b5d7981ca6fd448d2e328039f1aa9a1015950db055`.
- Reproduce: `pytest -q tests/test_sphere_polar_graph_update.py tests/test_atom_update_dispatch.py`.
- Complete job archive and verified per-file hashes:
  ignored `output/sphere-capturable-update/` in the sphere-baseline worktree.

The frozen performance source is `c10c435791ea77c6028221aa5bc70eb666b67287`.
Its new live-geometry rejection tests are run before measurements in the
performance job. The independent performance results below qualify the research Graph comparison;
public dispatch and public optimizer Graph support remain unchanged.

CPU source checkpoint c10c435: 1,354 passed, 2,395 CUDA/DB skips and 18 warnings,
26.03 seconds. This uses the repository's main `.venv` with `PYTHONPATH=src`.
Earlier system-Python invocations could not import the package and then picked
up an unrelated installed `tests` package; those were invalid environment runs,
not numerical failures or GPU validation. Offline wheel/sdist build passed via
cached `uv run --offline --no-project --with build --with hatchling python -m
build --no-isolation`. Ruff 0.16.3 checks pass after import-spacing cleanup;
that cleanup changes no numerical execution.


## Two independent complete-step measurements

NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, FP32 IEEE, B32,
5% atoms. Width3 uses W64/patch32; width8 retains bounded H/G chunk4096.
The second job reverses all four cases and all five modes. Every measurement
passes full-shape FP64 Y/dX/all dP gates both before and after 24 evolving steps.
Initial parameter/X/target/sites hashes agree across both jobs. Both upfront
regressions pass45 tests with zero skips; samples are21 timings per case, not
21 independent runs.

| N,width | public eager ms | research eager ms | research Graph ms | Graph allocated MiB | Graph reserved MiB |
| --- | --- | --- | --- | --- | --- |
| 1024,3 | 11.470 / 11.326 | 6.960 / 6.887 | 5.192 / 4.980 | 105.662 | 170 |
| 2048,3 | 30.516 / 29.783 | 27.085 / 26.536 | 25.915 / 24.960 | 324.259 | 398 |
| 1024,8 | 18.939 / 18.249 | 13.976 / 13.931 | 10.194 / 9.899 | 148.833 | 206 |
| 2048,8 | 102.867 / 101.813 | 98.388 / 97.247 | 94.836 / 93.127 | 299.432 | 390 |

Graph improves over the same research eager pipeline by25–29% atN1024 and
3.6–5.9% atN2048 in both independent jobs. It increases allocated memory by
roughly15–17MiB and reserved memory by58–64MiB relative to research eager.
Dense Graph remains substantially faster: about0.085–0.088ms atN1024 and
0.592ms atN2048, allocated32.877/81.502MiB. Allocator peaks do not establish
physical cache residency or total GPU process usage. Public eager differences
include synchronous guards and Torch coordinate handling; they are not a
measurement of isolated CUDA updater speed. A separate public-CUDA-selector
cohort keeps those wrapper guards to measure that question.

- Primary job: `l4job-2fe0c5582fb048ca84e16fbbb4296f05`.
- Primary source SHA256: `f844e0a5e23293675f641f07a2eceee011628b6cc93421b35e0ecfc8bf19420b`.
- Primary result SHA256: `f75de61e1634d96e6764022668cb654b75102f99ad39798392d3896dbb4b74c6`.
- Reverse job: `l4job-6e77960c0f49471a80817592467c8ad9`.
- Reverse source SHA256: `aeb6dcaea07270cb2d4f1ad4a41bbce0c7ecae04e3693c04e3d33e006cf8e184`.
- Reverse result SHA256: `8b567b5a5d3ae983ebd319fef91fc108f98bf3c958a69bd53124bb30941883e7`.

The full source-file inventories agree between these two jobs; only the
explicitly reversed driver differs. Curated medians/oracle proofs are in the
adjacent summary JSON; raw timing arrays and full verified job archives remain
in ignored output and the shared pool. Public CUDA selection and composed-plan gates/results are recorded below.


## Isolating the CUDA update inside public CSTOptimizer

The public-CUDA eager mode retains the same public gradient projection,
synchronous finite checks, AdamW proposal and vector-state transport. Only the
common update selector changes from the Torch reference to the CUDA Sphere
plan. Independent 20-step tests compare parameters/moments/counters and both
activity modes; non-finite gradients still reject before base mutation.
Both upfront regressions pass50 tests with zero skips.

| N,width | public Torch update ms | public CUDA update ms | allocated MiB |
| --- | --- | --- | --- |
| 1024,3 | 11.344 / 11.626 | 9.566 / 9.527 | 88.486 |
| 2048,3 | 30.650 / 29.738 | 29.082 / 28.721 | 308.008 |
| 1024,8 | 18.164 / 18.045 | 16.664 / 16.305 | 131.933 |
| 2048,8 | 102.204 (one job) | 100.608 (one job) | 283.854 |

The first three cases qualify above3% in both independent jobs; allocated peaks
are identical. N2048,width8 improves only1.6% in the primary job and was not
selected for an independent inverse run. The second job reverses the three
predeclared qualified cases and both modes; no thresholds were relaxed.
This isolates the updater gain from the removal of public guards in research
Graph execution. Public optimizer CUDA Graph capture remains unsupported.

- Public primary: `l4job-f475fa84222e4a439a231bff021e122e`.
- Source SHA256: `27765132084c47a2a926308261297c23ff254695b184a67dfab44f0a8ff83f18`.
- Result SHA256: `0ef2aa75d635c51b2fb0c7a2f71c859f8c7dd2a9f61f98684d2d0f8575a1c2ad`.
- Public inverse: `l4job-82df8ce0e1bb4b609533f84658fc09b3`.
- Source SHA256: `3ed0afadabe55f849e7ad7a09eefd2eb8729224d51f374a0de399406a0095dc8`.
- Result SHA256: `74e88c94a7fef9550e022e3fd0a1033070eab93d0dc4bd5fa61bec84cf218b6a`.

All1512 source entries match except the declared driver. Full initial parameter,
X,target and site hashes match across corresponding independent cases. Every
large measurement passes full-shape FP64 Y/dX/all dP initially and after training.

## Composed Linear and update plans

On merged preparation PR81, width3 uses the fused-preparation W64/patch32 plan
and width8 retains the best bounded H/G4096 plan. Both the old-plan/public-Torch
baseline and candidate-plan/public-CUDA use the same public CSTLinear binding.
The research Graph retains its explicitly stated guard boundary.

Compound gate `l4job-b2133f880ec942c8bb564fbc59c90378` passes72 tests without
skips, including20 public and20 Graph trajectories against independent Torch
factored execution. Source SHA256
`05fb1d13aceb603a41280014d9fd254a8c3fc4a750af7b54dce51ce912d22c1f`,
result SHA256
`9c26048ed2a1b52e3ef8cadb2442bc7be0cd42adad366a1a599b64dde9401ad2`.
Frozen source checkpoint: c0efcbf372ddecdea31a767fbfab7e14ac603ae6.

| N,width | public old plan ms | public compound ms | research Graph ms | public allocated MiB | Graph allocated MiB |
| --- | --- | --- | --- | --- | --- |
| 1024,3 | 11.291 / 11.291 | 7.927 / 7.949 | 4.784 / 4.729 | 88.486 | 105.662 |
| 2048,3 | 29.657 / 29.989 | 27.022 / 27.368 | 24.424 / 24.489 | 307.209 | 323.585 |
| 1024,8 | 18.033 / 18.023 | 16.275 / 16.157 | 9.865 / 9.928 | 131.933 | 148.833 |
| 2048,8 | 101.519 (one job) | 99.900 (one job) | 93.465 (one job) | 283.854 | 299.432 |

Primary compound job `l4job-05854f92ac3246a68228b405eb33d968` passes52
upfront regressions and all16 full-step measurements with matching initial/
shared fixture hashes and full initial/updated FP64 oracle gates. Source
SHA256 `fcc7cebb91ef4944e25e66f39fead921772e74712ca165fee43de61c938cb3bd`,
result SHA256 `725c9fb00de349bb66382c8cd864cd7f677dc953587feee01fdde3352354a295`.

The first three public compound cases qualify above3% in both independent
runs: about29.6–29.8% atN1024,width3,8.7–8.9% atN2048,width3, and9.8–10.4%
atN1024,width8. N2048,width8 is again only1.6% and remains a one-job result;
it was not selected for independent inverse. Allocated peaks are unchanged
atN1024 and slightly lower (0.799MiB) atN2048,width3. Graph allocated peaks
are higher than public eager and are reported separately; Dense Graph remains
much faster, around0.085ms atN1024 and0.594ms atN2048. Public dispatch stays
unchanged and no isolated kernel gain is inferred from public/Graph ratios.

Qualified compound inverse `l4job-6c7cb7ea1a23405188ed0ff0e02f71ef` passes52
upfront tests and all12 predeclared complete-step measurements with initial/
updated full-shape FP64 oracle gates. Source SHA256
`46aafdd10eb34cf584584a976932f392d7e4159740709106d40b2d3882a2d382`,
result SHA256
`07503e523189440d96858f70f891f664437455f77353a27cf507814cc1225166`.
Corresponding initial parameter/X/target/site hashes agree; source inventories
match except the explicit inverse driver. The second job reverses the three
qualified cases and all four modes. Graph comparisons retain their research
wrapper boundary; N2048,width8 compound Graph is one job, with the earlier
unchanged blocked Graph plan's separate independent evidence above.

Final composed-source CPU check: 1,355 passed, 2,421 CUDA/DB skips,18 warnings,
26.20 seconds. Ruff0.16.3 and offline wheel/sdist build pass. Optimizer core is
unchanged; the API additions are research algorithms and independent benchmark
selection/testing paths.
