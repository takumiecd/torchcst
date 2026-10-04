# Pull-request integration of validated local-product research

2026-10-05: the user explicitly requested integration into main after database
preservation and research dispatcher generation. The user clarified that remote
integration must use a GitHub PR and that direct pushes to main are prohibited.
The prematurely prepared local merge `97c97699` was preserved on
`codex/local-product-main-pr`; local main was restored to `origin/main` at
`6eee7fae`. The validated changes are published only through this PR branch.
After GitHub merges the PR, local main is synchronized by fast-forward only.
AGENTS.md records the PR requirement for subsequent work.

The PR contains the small normalized/shared-width PolarAmpWidth product
algorithms, per-atom local/saved H and optional saved G, persistent execution
layout, existing Linear benchmark Cases/Plans, revision-3 database projection,
case-scoped dispatch generation requests, tests and research records. Algorithms
remain explicit research recipes in the benchmark Registry; integration into
main does not make them the public CSTLinear default. Persistent execution still
uses model-owned state in the existing research runner. Case-scoped exact
candidates are not an automatic selector from current rho mixtures.

## Integration validation and compatibility repair

The first full suite found two failures in existing normalized Strip CPU tests.
Both also failed in a frozen checkout of pre-merge main `6eee7fae` under the same
Python3.13.8 / Torch2.13.0 runtime, proving this was not introduced by the merge.
At exactly clamped width endpoints, this runtime's scalar clamp produced zero
gradient and tensor-bound clamp produced a different tie gradient. The existing
CUDA kernels explicitly retain the interior width derivative at both endpoints
(`log >= LO && log <= HI`).

The normalized CPU reference and independent sampled-site oracle now use
explicit endpoint choices in their stored dtype to preserve that inclusive
contract, with zero derivative outside. No tolerance was relaxed. An independent
one-sided finite-difference test specifies the upper-end interior derivative and
checks the zero exterior derivative. No CUDA kernel or product/polar math was
changed by this repair.

After repair, the candidate's complete suite is **778 passed / 302 skipped** (20.31s).
Focused public/dispatch checks are42 passed /12 skipped. Skips include unavailable
CUDA hardware and dedicated PostgreSQL tests; they are not counted as passes.
Prior validation remains the95 final GPU-host checks /24 full-shape independent
FP64 comparisons /20-update public optimizer checks, and214 CPU/real-PostgreSQL
storage/selection/submission checks documented at their validated checkpoints.
This integration allocates no GPU and claims no new GPU timing measurement.
Ruff and whitespace checks pass for the compatibility repair.

Every local-product backend Python file matches the research branch byte-for-byte.
The optimized kernels remain those measured at `9dffad67`; historical observation
source IDs and recorded versions are preserved rather than relabeled as main.

## Evidence and rollout status

The ten completed artifacts already live in the configured PostgreSQL database,
with byte-exact export and idempotent reimport validation. Main integration makes
their adapter and generation code available here; it makes no further DB writes.
Their [registration record](20261005-database-and-dispatch.md) and
[structured identities](20261005-database-and-dispatch.json) remain authoritative.

Copied27 generated-candidate/receipt/log files into main's ignored
`benchmarks/database/evidence/local-product-20261005/` and verified every SHA256
against the research checkout. Thus the saved-dataset reproduction commands work
from main too. Original experiment raw outputs, frozen source archives and pool
receipts remain preserved in recovery archives and host-wide pool directories.
No ignored PostgreSQL test-server data or credentials were copied or committed.
Worktree recovery paths and the archive procedure are recorded in
[the cleanup record](20261005-worktree-cleanup.md).

Main integration logs, pre-merge frozen checkout, initial failures, final suite
and copy-hash manifest are ignored under
`benchmarks/database/evidence/main-integration-20261005/`.
Remote integration uses a GitHub PR; no direct main push or production-default
selection is part of this checkpoint.

```bash
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:. python -m pytest -q tests/test_normalized_strip_public.py tests/test_cuda_dispatch.py
PYTHONPATH=src:. python -m benchmarks.dispatch --request benchmarks/dispatch/requests/local-product-l4-20261005/64-middle-baseline-peak.json --dataset benchmarks/database/evidence/local-product-20261005/generated/64-middle-baseline-peak/dataset.json --output output/local-product-middle-offline
```
