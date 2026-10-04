# Research worktree recovery and PR handoff

2026-10-05: the user requested cleaning up research worktrees before resuming and
clarified that main integration must happen on GitHub through a PR. Direct main
pushes are prohibited. No direct main push was performed. The local tested merge
`97c97699` is retained in `codex/local-product-main-pr`; the premature local main
advance was restored before publishing this feature branch.

The four inactive research checkouts are archived after preserving their branch
heads, clean/dirty status and all source/evidence files, including non-ignored
untracked files. Archives exclude only `.git`, the rebuildable `.venv` and Python/
pytest/Ruff caches. Each regular file is verified by SHA256 against the archive;
symlinks are verified by their targets. Git branches and commits remain available.
The managed local-product checkout additionally uses Codex's recoverable archive.
Unrelated active chats/projects are not modified.

| Branch | Retained head | Additional local files preserved |
|---|---|---|
|`codex/cst-search-dataflow-20261003`|`3c429c0a`|ignored experiment evidence|
|`codex/cst-search-preparation-20261003`|`ac0fbad7`|`polar_local_width_rank_review.py`, contract JSON, ignored evidence|
|`codex/cst-search-window-20261003`|`68fdf888`|contract JSON, ignored evidence|
|`codex/local-product-hybrid`|`e211ecb3`|complete raw experiment/DB evidence and frozen sources|

Recovery files remain ignored under the primary checkout:

```text
benchmarks/cuda/linear/evidence/archived-worktrees/20261005-pr-handoff/
    manifest.json
    cst-search-dataflow-20261003.tar.gz
    cst-search-preparation-20261003.tar.gz
    cst-search-window-20261003.tar.gz
    local-product-hybrid.tar.gz
```

The manifest stores full commit IDs, original paths/status, per-file hashes and
archive hashes. Generated dispatcher files, import receipts and primary full-suite
logs also remain directly available under the primary checkout's ignored evidence.
No raw archives, logs, dependency environments or credentials enter the PR.

To recover an old manual research checkout, create a worktree on its retained
branch and extract the corresponding archive there. For example:

```bash
git worktree add /tmp/torchcst-preparation codex/cst-search-preparation-20261003
tar -xzf benchmarks/cuda/linear/evidence/archived-worktrees/20261005-pr-handoff/cst-search-preparation-20261003.tar.gz -C /tmp/torchcst-preparation
```

Codex's archived attachment can restore the managed local-product checkout.
Extracting its archive after restoration recovers the ignored evidence too.
Recreate `.venv` from the repository's dependency declarations when resuming.
