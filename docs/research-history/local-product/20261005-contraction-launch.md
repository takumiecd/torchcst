# 2026-10-05: local contractions, launch tuning and bounded unrolling

Development branch: `codex/local-product-contraction`, based on main `bcf69e5b`
(GitHub PR #22). Continue the existing small Linear runner; no GEMM implementation
or alternative benchmark runner is introduced. Sigma evolves during every step.

## Findings

L4, batch32, FP32, TF32 disabled, about5% atoms, production PolarAmpWidth update
with fused AdamW proposal/polar policy. The comparison is the uninstrumented
complete training step and capture/replay peak allocated memory. Diagnostics use
a different instrumented graph. Allocator reserved and GPU process usage are
separate; process usage remains unmeasured.

At N128 early/mixed widths, processing32 atoms with4 contraction warps improves
the saved-G control from about181us to about160us (roughly12%). Without savedG,
about192us becomes about167–168us (roughly13%). Allocated peaks stay509,952 and
417,280 bytes respectively; both reserve6MiB. The faster no-G candidate also uses
less allocated memory than the previous saved-G control while beating its time.
The recipe changes the loop granularity and launch only, retaining canonical
atom IDs, full normalization, task VJP and persistent layouts.

A bounded eight-iteration unroll improves ordinary initial-sigma3 N64 from
about85.5us to78.5us in its first measurement; a separate L4 reproduces85.1→78.1us, without increasing the152,064-byte
allocated peak. Only the middle band uses it, and the **current actual support
span** must be at most8. Other spans use the original contraction. This bounds
instructions, not sigma; widths continue to change during training.

N64 mixed/narrow cases do not benefit from atom32, and retain atom16. Support
vectorization is a negative result: a batch×atom×support temporary increases
register pressure and sometimes spills, slowing the complete step despite
unchanged PyTorch allocation peaks. At ordinary N64 sigma3, forward registers
rise56→168 with2 spill slots. Wider runtime guards and4 warps do not rescue it.
Parameter VJP at4 warps removes N128's4 reported spill slots but is slightly
slower overall; eliminating spills alone is not a speed criterion.

## Correctness and repetition

Independent full-shape FP64 scalar gates verify Y, dX and all atom gradients.
Captured20-update checks compare production CSTOptimizer parameters, moments,
step, widths and task gradients. Sliced domains and retained backward are checked.
Atom32 additionally passes deliberate singleton migration, capacity overflow,
width expansion and outstanding-forward topology checks in CUDA Graph replay.

Two paired repeat **benchmark executions** recreate model/optimizer/input state
in fresh runner worker processes. They share one pool job/VM, have distinct
execution UUIDs, and start from identical parameter/input hashes. These are two
independent benchmark executions, not21 independent executions from21 time
samples. The N128 improvements are also seen in the earlier separate pool job.

Raw files and frozen drivers remain under the shared pool's `jobs/JOB_ID/` and
this worktree's ignored `benchmarks/cuda/linear/evidence/contraction-launch-20261005/`.
[Machine measurements](20261005-contraction-launch.json) retain job IDs, source
snapshot/result-archive hashes, runtime, oracle errors and per-case medians/peaks.
Reproduction uses the job's `source.tar.gz` and `__pool_driver__.py`; source
snapshots include uncommitted research code and are identified by SHA256, not
misrepresented as the base commit. Runner metadata records the working-tree
label separately from `commit=unrecorded`.

## Workflow and candidate status

[Kernel development workflow](../../kernel-development.ja.md) documents source,
Plan/Case, independent validation, owned pool execution, exact DB preservation,
research selector generation and GitHub PR integration. No direct push to main.

Completed positive and negative observations are appended to the existing
PostgreSQL store with owned-pool provenance, byte-identical restoration and
idempotent reimport checks. New generation requests use the exact source/runtime,
filter each case separately, and require at least2 independent benchmark
executions. Speed and control-peak policies are separate. Control peak is a
comparison policy, not a user-specified memory ceiling. Initial rho distributions
still are not runtime selector keys, so these remain case-scoped research
candidates; the public CSTLinear default is unchanged.

`plans-local-tuned.json` is the single catalog for the new recipes, including
explicit negative alternatives for evidence reconstruction. `local-tuned-*`
cases select the four control/atom32 comparison plans. Frozen exploratory
catalogs remain recoverable from the research branch and pool snapshots.
Combining the unroll with atom32/4-warps remains a future measured experiment.

A further candidate is to group narrow/middle atoms by the tile containing their
first supported site, separately for output and dX. If the actual target-axis
span is at most one tile width, each tile needs only its own and the preceding
owner bucket. Longer/rounding-degenerate spans need a global fallback bucket.
This could avoid scanning every middle-band atom block for every output tile;
additional bucket slack, maintenance and peak memory must be measured first.


## Final verification

CPU full suite:779 passed /324 skipped (18 existing TorchScript deprecation
warnings). Final independent L4 validation:23 passed, covering all new launch,
vector and unroll routes,20-update public optimizer comparisons, sliced retained
backward and atom32 migration/overflow/topology. The separate final measurement
passes12 full-shape FP64 gates and12 runner gates plus2 matched dense steps.
All173 Python source hashes in that measurement match the proposed PR source.
Its source/environment/protocol/case IDs match the repeat cohort exactly.

The combined gate `l4job-af4f8b3c817145ca97dc9d0767ed7c72` was interrupted by
Colab CLI completion transport timeout after1340seconds. No verified numerical
or performance result was recovered, so none is claimed. The pool stopped the
owned runtime; documented recovery succeeded. Replacements
`l4job-106d362646b5451fb2e4599268d3abd3` (validation) and
`l4job-2b2a91c4e67248288f9aadcd310d839d` (measurement) both succeeded on a fresh L4.
The second pool supervisor exited successfully and stopped its owned runtime.


## Final storage and case-scoped candidates

Thirty completed runner artifacts, including negative alternatives, are appended
with exact byte restoration and idempotent provenance checks. DB counts move
8→24 plans,14→44 runs,126→500 records and798→3184 metrics; the same7 cases are
shared by content identity. The interrupted transport job is not performance
evidence. [Storage receipt](20261005-tuned-database.json) records the30 run IDs.

Eight [candidate registrations](20261005-tuned-dispatch.json) are generated from
source `8429ffcb00c0300fb885e291d09a9b40279bc9e04950361f3bd653eb011bf652`,
with observed Torch2.11.0+cu130/Triton3.6.0 and minimum2 independent executions.
Median-of-run medians and maximum complete-step allocated peaks:

| Case | Speed route | Step(us) | Peak allocated(bytes) | Executions |
|---|---|---:|---:|---:|
|64 middle|prepared bands, atom16|66.387|152,064|3|
|64 narrow-only|prepared bands, atom16|44.369|152,064|3|
|64 ordinary initial-sigma3|bounded unroll, atom16, noG|78.305|152,064|2|
|128 early|atom32/4-warps, savedG|160.224|509,952|3|

The N128 control-peak policy selects atom32/4-warps withoutG:
167.486us,417,280 bytes,3 executions. Other control-peak winners match speed.
All candidates reserve6MiB in the observed runs. Exact lookup and unobserved
batch fallback pass structural checks; executable load rejects the local
Torch2.13 runtime mismatch. These checks do not claim GPU execution on the
local host. Actual L4 candidate kernels were measured separately.

Only the consolidated catalog and four stable case declarations enter the PR.
Exploratory catalogs are retained at research checkpoint `afc03a81`; the173
measured Python file hashes are unchanged by this configuration/document cleanup.
