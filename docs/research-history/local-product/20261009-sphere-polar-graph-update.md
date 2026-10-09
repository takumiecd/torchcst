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
performance job. Performance and memory results are pending; no speed claim or
public dispatch adoption is made at this checkpoint.

CPU source checkpoint c10c435: 1,354 passed, 2,395 CUDA/DB skips and 18 warnings,
26.03 seconds. This uses the repository's main `.venv` with `PYTHONPATH=src`.
Earlier system-Python invocations could not import the package and then picked
up an unrelated installed `tests` package; those were invalid environment runs,
not numerical failures or GPU validation. Offline wheel/sdist build passed via
cached `uv run --offline --no-project --with build --with hatchling python -m
build --no-isolation`. Ruff 0.16.3 checks pass after import-spacing cleanup;
that cleanup changes no numerical execution.
