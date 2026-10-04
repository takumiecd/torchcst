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
about85.5us to78.5us in its first measurement, without increasing the152,064-byte
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
