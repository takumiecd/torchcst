# Local product observation storage and research dispatcher candidates

2026-10-05, branch `codex/local-product-hybrid`, kernel checkpoint `9dffad67`.
The user requested ending this optimization round, preserving its observations
in the existing database and registering research dispatch candidates. No CUDA
kernel, mathematical contract or production Registry changes in this checkpoint.
GPU times remain the measurements in the
[support preparation / backward G record](20261004-support-preparation-and-backward-g.md).

## Adapter and selection boundary

The existing complete-step adapter retains normalized Strip revision 2 and its
identities. Local product uses revision 3 with the benchmark-local Registry.
Reconstruct the shared-width normalized Triweight / PolarAmpWidth Operator from
the Case, including width bounds below spacing; persist the complete initial rho
mixture, source hashes and all raw diagnostics. Validate Y/dX/all atom-gradient/
polar-update errors, all performance workers, ordinary dense reference, precision,
optimizer and initialization identities. `polar_update` and the production polar
optimizer policy are explicit protocol fields. No full-shape certification or
unmeasured GPU-process memory is invented.

The measured source identifier `de1a8c80-contraction-workingtree` is a label,
not a full Git SHA. Revision 3 retains it as `recorded_source_label`, sets commit
to `unrecorded`, and uses recorded source hashes to separate snapshot content.
The original JSON bytes are preserved. No snapshot is relabeled as the final
Git commit after measurement.

Generation reconstructs local parameter_dim=4, both one-dimensional charts,
recorded width bounds, shape, batch, atom count, device and precision. CLI and
offline generation use the benchmark Registry; production registration remains
unchanged. The persistent execution route still requires model-owned layout
state, as in the existing Linear runner. These artifacts are research candidates,
not newly wired public CSTLinear implementations.

Initial mixtures are not visible in the runtime exact key. N64 mixed and
narrow-only can have identical runtime conditions; generation keeps each Case
separate and rejects an ambiguous combined dataset. No arbitrary merging or
learned-stage routing is introduced.

## Actual database preservation

The existing configured PostgreSQL database had schema migration 2, verified
before writes. No migration or role changes were needed. A single transaction
appended the ten original completed runner artifacts from:

- Initial/negative batch `l4job-5b39198a2dbd4671bd2b6941faf96fa3`: six Cases,
  four Plans each, including the naive saved-G regressions.
- Final batch `l4job-b37da59e240841bf952bd4a8f6c108aa`: four Cases, six Plans each.

The batch added 10 runs, 10 projections, 48 Plan measurements, 48 correctness
workers and 10 dense references: **106 worker observations / 674 metrics**.
Database totals changed from 2 Plans / 1 Case / 4 runs / 20 records / 124 metrics
to **8 Plans / 7 Cases / 14 runs / 126 records / 798 metrics**. Original existing
observations remain unchanged.

Every imported run was exported and compared byte-for-byte with its input.
Every run was reimported with the same caller provenance and added neither a
run nor a projection. Pool job IDs, frozen-source and verified-result archive
SHA256s accompany each observation's caller provenance. The compilation-failure
pool receipt and the fixed-state preparation probe are not complete-step schema
artifacts and remain in ignored evidence; no fake benchmark workers were made
for them.

Initial snapshot source ID:
`0331bd639c85c964a94bdab1b2d77ad726d1f345dbb6bfa3c583bc99b12f29cc`.
Final snapshot source ID:
`69b560101ea47bd3104dd6e022d33e0e0cf6ee9b65810ea46cab75e81da9ac1b`.
The [tracked receipt/registration summary](20261005-database-and-dispatch.json)
contains all run, Case, source, dataset and dispatch-content IDs.

## Eight case-scoped registrations

Read fixed, read-only database snapshots for the final source, filtered by Case,
environment and protocol. Preserve all eligible candidate observations, not
only the winning Plan. Generate two policies per Case:

- `speed`: lowest measured complete-step Graph time.
- `baseline-peak`: speed with allocated peak capped at the previous control's
  measured peak. This is an explicit comparison policy, not a user-imposed
  numerical memory ceiling.

| Case | speed candidate / us / allocated bytes | baseline-peak candidate / us / allocated bytes |
|---|---|---|
|N64 mixed|prep+band / 66.157 / 152064|prep+band / 66.157 / 152064|
|N64 narrow-only|prep+band / 43.992 / 152064|prep+band / 43.992 / 152064|
|N64 initial sigma3|prep+band+G / 84.518 / 178176|prep+band / 84.836 / 152064|
|N128 early|prep+band+G / 180.159 / 509952|prep+band / 192.010 / 417280|

`prep+band` is catalog alias `persistent-supportprep-band`, route
`persistent_supportprep_band`. The G candidate is alias
`persistent-supportprep-g`, route `persistent_supportprep_g`.
All artifacts fall back to measured control `hybrid-persistent-mid4`.
Peaks include full-step Graph capture/replay; reserved is 6MiB for these CST
candidates, and whole-process GPU usage is unmeasured. The sigma3 speed choice
is a 0.318us difference from noG, not evidence of G reuse (no wide G lanes).
Keep the lower-memory policy as the N64 development choice.

Each Case has **one independent final-source run**, with 21 timing samples.
Requests explicitly set `min_runs=1`; this registers measured development
candidates without claiming repeated-run significance or universal winners.
Artifact runtime versions are the recorded Torch2.11.0+cu130/Triton3.6.0, not
the generator's installed versions. Structural validation and offline replay
are checked; the new selector wrapper was not timed on GPU or promoted to a
production default. Selection from current rho distributions remains future work.

## Verification and reproduction

**214 tests pass**, including actual isolated PostgreSQL import/export,
idempotence, rollback, append-only enforcement, existing normalized paths,
submissions/automation and local-product dataset/selector generation. The new
synthetic tests prove speed versus baseline-peak selection, source-label handling,
width-contract rejection, optimizer mismatch rejection, local fallback, offline
CLI replay and Case collisions. Synthetic tests are not GPU observations.
Ruff and whitespace checks pass. The initial local test server accidentally used
SQL_ASCII; its text decoding prevented migration-history comparison. Recreated
the owned server with UTF8, reran all checks, and stopped both owned servers.
Their data/logs remain on disk; the configured observation database is unrelated
to these disposable test databases.

Tracked requests are under
[`benchmarks/dispatch/requests/local-product-l4-20261005/`](../../../benchmarks/dispatch/requests/local-product-l4-20261005/).
Full generated dataset/leaderboard/dispatch JSONs, import receipts, logs and
registration summaries remain ignored in
`benchmarks/database/evidence/local-product-20261005/`.
Existing measured raw source/results stay in the prior experiment's evidence.
No GPU resource was allocated for this storage/selection checkpoint.

With a configured `DATABASE_URL` (kept private; `.env` is not auto-loaded):

```bash
PYTHONPATH=src:. python -m benchmarks.dispatch --request benchmarks/dispatch/requests/local-product-l4-20261005/64-middle-baseline-peak.json --output output/local-product-middle
PYTHONPATH=src:. python -m benchmarks.dispatch --request benchmarks/dispatch/requests/local-product-l4-20261005/64-middle-baseline-peak.json --dataset benchmarks/database/evidence/local-product-20261005/generated/64-middle-baseline-peak/dataset.json --output output/local-product-middle-offline
PYTHONPATH=src:. python -m benchmarks.database records --adapter-revision 3 --gpu 'NVIDIA L4' --kind measure --limit 100
PYTHONPATH=src:. python -m benchmarks.database export-run c00a922935a26329c8bee5aceab3bcb417528b194c63ce72091565fce6164566 output/local-product-middle-original.json
PYTHONPATH=src:. TORCHCST_TEST_DATABASE_URL='host=/tmp port=55443 dbname=postgres' python -m pytest -q tests/test_benchmark_database_adapter.py tests/test_benchmark_database_postgres.py tests/test_dispatch_generation.py tests/test_dispatch_generation_postgres.py tests/test_dispatch_selectors.py tests/test_local_product_database.py tests/test_benchmark_submissions.py tests/test_benchmark_submission_postgres.py tests/test_benchmark_automation.py
```

The final command requires an owned disposable UTF8 PostgreSQL test server.
`export-run` never overwrites an existing file. To reimport a preserved artifact,
use `Database.import_linear(raw, provenance=receipt["provenance"])` with the
tracked original receipt provenance; using a different origin is rejected.
