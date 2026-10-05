# Repository instructions for agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing this repository.
It is the canonical contributor workflow. When using the owner's GPUs or research
worktrees, also read [the research operating rules](docs/research-operations.ja.md).
For CUDA changes, read [the kernel development guide](docs/kernel-development.ja.md).

- Integrate through a named `codex/` branch and GitHub PR; never push directly
  to `main`. Fast-forward local `main` only after GitHub merges the PR.
- Before GPU work, read the research operating rules and the applicable resource
  skill. Allocation and shared-pool ownership limits
  remain in force; this file does not grant new resource authorization.
- Preserve commits, uncommitted work and needed ignored evidence before worktree
  cleanup. Follow the evidence and recovery rules in
  [the research operating rules](docs/research-operations.ja.md).
- `README.md` defines the public API. Respect the declaration/execution boundaries
  in [the backend layout](src/torchcst/_backends/README.md); keep benchmark
  tooling outside the runtime package and validate mathematical contracts.
