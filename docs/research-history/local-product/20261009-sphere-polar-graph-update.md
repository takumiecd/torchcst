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
in ignored output and the shared pool. Public-optimizer CUDA selection and
compound fused-preparation tests are pending their own GPU gates.
