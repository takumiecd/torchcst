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
