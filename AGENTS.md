# Main integration must use pull requests

- The user explicitly prohibits direct pushes to `main` (2026-10-05).
- Publish changes on a named `codex/` branch, create a GitHub pull request, and
  merge through the pull request after the required validation passes.
- Do not substitute a local merge into `main` for remote PR integration. After
  GitHub merges the PR, fetch and fast-forward the local `main` to `origin/main`.
- Preserve commits, uncommitted files and needed ignored evidence before
  archiving or removing research worktrees. Keep recovery paths in research notes.

# Shared Colab L4 experiments

For Colab L4 research experiments, read and use
[the shared pool skill](tools/colab-l4-pool/SKILL.md). It is also installed at
`~/.codex/skills/colab-l4-pool` on the user's local host.

- All local chats/worktrees submit to the default host-wide queue. Do not
  create a per-agent live queue with `--state-root`.
- One orchestrator owns `serve`; other agents submit jobs and wait for results.
  Default to one L4. The orchestrator may select two or three when warranted
  within the user's authorized allocation scope.
- Do not directly upload, execute, restart or stop the pool's `cst-pool-*`
  sessions through the CLI or notebook UI. The pool serializes complete
  experiments and manages isolated subprocesses, result retrieval and cleanup.
- A wait timeout does not release a runtime. Use `status` and the documented
  recovery path for interrupted work; do not edit the database or slot records.

Other GPUs and sessions outside this pool retain their existing workflows.
This queue coordinates one host; another computer must not operate its VMs.

# Cross-generation experiment scope

- The user authorized Blackwell/G4 and other GPU generations on 2026-10-01.
  Keep the original L4 performance objective and record other devices separately.
- G4 use is cost constrained. The orchestrator alone selects and allocates short
  comparison runs for promising, validated candidates. Agents must not provision
  G4 independently or run broad G4 parameter sweeps.
- Measure a complete dense training step on each comparison GPU under the same
  model, batch, dtype, precision settings and optimizer contract. Preserve kernel
  normalization, sharp support/one-hot behavior, dX and all atom gradients.
- Record actual hardware and runtime versions, retrieve verified results and stop
  owned runtimes after the selected batch. A faster GPU is not evidence of an
  algorithmic improvement on L4.

# Complete-step memory objective

- The user added low memory consumption as a requirement on 2026-10-01.
  Evaluate time and memory together; retain normalization, support and gradient
  correctness. There is no user-specified numerical memory ceiling.
- Report peak allocated memory for the complete step including CUDA Graph
  capture. Distinguish allocated bytes, allocator reserved bytes and total GPU
  process usage; do not present a tensor budget as a measured GPU peak.
- Prefer bounded weight/weight-gradient scratch and buffer reuse. Keep new
  implementations on research branches until independent correctness checks
  and actual GPU time/peak measurements pass. Label sharp-only fixtures separately
  from the ordinary sigma-three performance objective.

# Research Git checkpoints

- Keep experiment worktrees on named `codex/` branches. Commit coherent source,
  tests and research notes at validated checkpoints; record negative results too.
- Keep generated logs, traces, tensors and frozen source copies in ignored
  `evidence/` directories. Preserve these files on disk; do not bulk-add them.
  Track concise results, job IDs, source hashes and reproduction commands in notes.
- Keep experimental alternatives on their research branches until validated for
  the main tree's mathematical and numerical contract.
- Before Colab submission, inspect `git status` and ensure raw evidence copies
  will not enter the source snapshot. Existing intentionally tracked evidence
  remains tracked; new raw output is ignored.
