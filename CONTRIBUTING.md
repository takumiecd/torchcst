# Contributing to torchcst

**English** | [日本語](CONTRIBUTING.ja.md)

We welcome code improvements and GPU benchmark results. Submit code through pull
requests and measurement results through this repository's issues. Submitting
measurement results does not require a fork, a submission branch, a PR or Neon
connection credentials.

## Adding code or kernels

Start development with this guide. For CUDA kernel implementation, registration
and measurement details, see the [kernel development guide](docs/kernel-development.md).

Choose branch names by the purpose of the change, rather than the author's or
tool's name. Use `kernel/<purpose>` for new kernels, kernel optimizations,
recipe/launch tuning and the associated validation and measurements. For other
changes, use an appropriate prefix such as `feat/<purpose>`, `fix/<purpose>`,
`refactor/<purpose>`, `docs/<purpose>`, `test/<purpose>` or `chore/<purpose>`.
Integrate all changes through GitHub PRs.

```text
Define the mathematical contract of the change
    → Implement on a kernel/ branch and register the Algorithm/Recipe
    → Prepare a Plan/Case; check declarations, an independent oracle and regression tests
    → Measure baseline, candidate and dense on a real GPU under matching conditions
    → Record correctness, complete-step time, peak memory and evidence
    → Review the PR, pass the required validation and merge on GitHub
```

### 1. Set up the development environment

Run these commands from the repository root. External contributors can open PRs
from a fork.

```bash
python -m pip install -e '.[dev]'
git switch -c kernel/my-kernel
python -m tools.kernel_dev test --suite cpu
```

On GPU hosts, use `python -m pip install -e '.[dev,cuda]'`. When running CPU
validation in an environment where CUDA is visible, set `CUDA_VISIBLE_DEVICES=''`.
Skipping tests that require a GPU or a live database does not complete GPU or
database validation.

### 2. Register the implementation and comparison conditions

Distinguish the mathematical KernelSpec from a CUDA Algorithm/Recipe. When
changing how an existing operation is computed, put the implementation in
`_backends/cuda/algorithms/`. The changes required for recipe tuning, new
Algorithms and new mathematical operations are listed in
[types of additions and files to change](docs/kernel-development.md#types-of-additions-and-files-to-change).
The runtime package in `src/torchcst/` must not depend on `benchmarks/` or `tests/`.

The authoritative public API contract is [README.md](README.md); the current
package boundaries are defined in the [backend layout](src/torchcst/_backends/README.md).
An operation is defined as the sum of kernels over atoms; matrix factorization is
one way to compute it. Keep atom counts and Parameter shapes fixed during
training, and store trainable values such as amplitude, bandwidth and center in
the atoms' Parameters. Separate Module state ownership, differentiation and
optimizer updates. Do not restore compatibility interfaces for the old store,
birth/death/merge, slot remapping or Pullback Adam.

Use a Plan catalog and Case format that the existing Linear runner can read.
The following example uses an already registered candidate and can run without a
GPU.

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans.json \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --candidate window512 \
  --output output/kernel-comparison
python -m tools.kernel_dev check \
  --plans output/kernel-comparison/plans.json \
  --case output/kernel-comparison/case.json
```

`prepare` saves comparison JSON containing the baseline, the specified candidate
and dense, together with hashes of the source files, in a new directory. Repeat
`--candidate` to include multiple candidates. It does not overwrite existing
output or automatically generate or register a new implementation. After
registering a new candidate, use the same procedure with its catalog and Case.

### 3. Validate correctness, performance and memory

Use an independent oracle to validate Y, dX, all atom gradients, boundaries,
support and normalization. When changing an optimizer, also compare Parameters,
moments and step counters. Check Graph replay and variable widths on a GPU, and
measure the complete training step without instrumentation. Compare time and
peak allocated/reserved memory, including capture/replay, under matching
conditions.

The [minimal runnable example](docs/kernel-development.md#try-the-workflow-with-a-registered-algorithm)
provides commands for CPU checks, GPU correctness, complete-step measurements
and checks before submission. The [research operating rules](docs/research-operations.md)
also apply when using the owner's shared GPUs. External contributors measuring
on their own GPUs do not need the owner's GPU accounts or database credentials.

### 4. Record the work and open a PR

Commit code, tests and concise research notes. Keep raw logs, traces, tensors and
source copies in ignored `output/` or `benchmarks/**/evidence/` directories.
Submit measurement results through the issue process in this guide; PRs can
reference submission issues or observation IDs. New Plans and result formats
are accepted only after the central registry/adapter supports them, so the code
support is reviewed first.

Use the PR template to describe the purpose and contract of the change,
validation performed and not yet performed, hardware/runtime, complete-step
time and peak memory, the number of independent runs, source/result hashes and
reproduction commands. The PR's **CPU validation** Actions run declaration
checks, CPU tests and wheel/sdist builds. GPU correctness and performance, and
live database behavior, require separate validation. Adoption by the public
dispatcher is also a separate decision.

Push a named branch and integrate through a GitHub PR; never push directly to
main. After GitHub merges the PR, fast-forward local main to `origin/main`.

## Contributing benchmark results

```text
Measure on your own GPU / Colab / a rented GPU
    → Write complete result JSON to output/
    → Check the format locally before submission
    → Attach the JSON to a torchcst issue and submit it
    → Central GitHub Actions check the format, internal consistency and submitter
    → Neon checks for duplicates and submission limits, then appends the original and observation
    → The issue receives the observation ID, submitter, category and points, and is closed
```

### 1. Measure a provided Case

Currently, submissions must be complete schema v1 JSON generated by
`benchmarks.cuda.linear.run` for CUDA GPUs. Run in an environment that satisfies
the Algorithm's requirements for the GPU. A Case describes inputs, the model and
measurement conditions; a Plan describes the Algorithm to execute and its
settings. See the [CUDA benchmark guide](benchmarks/cuda/linear/README.md) for
choosing a Case or Plan.

Example commands from the repository root:

```bash
python -m pip install -e '.[cuda,benchmark-db]'
python -m benchmarks.cuda.linear.run \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --output output/normalized-step.json
```

The runner records a run UUID, UTC start time, Plan, Case, GPU/CUDA/Torch/Triton
information, timing samples, errors and peak allocated/reserved memory including
CUDA Graph capture/replay. Allocated and reserved memory are different metrics.
This format does not measure total memory usage of the GPU process. A small
independent oracle for correctness and performance measurements on large inputs
cover different validation scopes.

`output/` is ignored by Git. Keep result JSON, logs, traces, tensors and source
snapshots locally without committing them. Complete FAIL results can also be
submitted as diagnostic observations.

### 2. Check the JSON before submission

```bash
python -m benchmarks.submissions check output/normalized-step.json
```

This check requires no GPU, GitHub login or database connection. Incomplete
results and unsupported formats cannot be submitted.

### 3. Attach the result to an issue and submit it

1. In this repository, select **Issues → New issue → Submit benchmark result**.
2. Drag one complete JSON file into the `Benchmark JSON` field to attach it.
3. Click **Submit new issue**. Attaching the file alone does not start processing.
4. Check the **Submit benchmark observation** Actions run and the reply on the issue.

General and certified contributors use the same form. Include only one link to
the attached JSON in the issue body. Each issue accepts one run's JSON, up to
5 MiB. Multiple Plans or timing samples still count as one submission. The
attachment and measurement information are public; do not include credentials.

### 4. Check the submission outcome

Central Actions retrieve the issue author's numeric ID and username through the
GitHub API. Usernames, categories and points written in the JSON are not used to
determine authorization. CPU checks validate the format, consistency with the
selected Plan/Case, snapshot hashes, the worker configuration of PASS results,
timing samples and medians, and memory consistency.

Only the storage step receives the Neon connection Secret. Contributors' code
is not executed centrally. The database checks submission limits and appends
the original result, observation and submitter in a single transaction, so
concurrent submissions cannot exceed the limit.

| Outcome | What happens next |
| --- | --- |
| Saved successfully | The issue receives the observation ID, submitter, category and points, and is closed. |
| Duplicate result | The existing observation is returned; no new observation is created and no additional submission allowance is used. |
| Invalid format or other problems | The issue remains open. Before registration, the author can edit its body to trigger another check. |
| General submission limit reached | Processing can resume after the UTC date changes, through a body edit or an administrator's rerun. |

Submit a new issue for a different measurement. Observations already registered
from an issue are not replaced. If an administrator reruns Actions, the original
issue author remains the submitter.

## Submission limits and certification

| Category | Limit on new runs | Trust points |
| --- | --- | --- |
| General | 10 per UTC day per GitHub numeric user ID | 1 |
| Certified | No submission count limit | Default 10; administrators can set an individual value |

General contributors need no advance registration. To certify a contributor, an
administrator records their numeric GitHub ID, display name and reason in the
[public configuration](.github/benchmark-submissions.json). Identity is checked
by numeric ID even if the display name changes. Certified submissions use the
same format and size checks. See the [administrator instructions](docs/benchmark-contributions.ja.md#管理者による認定)
for adding or changing certification.

Counts use the UTC date of database receipt, rather than the measurement date.
A complete FAIL result also counts as one submission. Resubmitting identical
JSON, even through another issue or account, creates no duplicate observation
and preserves attribution to the first submitter. Differences only in JSON
whitespace or key order still represent the same observation. Independent
measurements are appended with different run UUIDs; conflicting results with
the same UUID are rejected.

Trust points are recorded for each submission using the settings in effect at
receipt. There are currently no automatic point increments based on submission
count and no ranking weights based on points. Certification or configuration
changes do not rewrite past observation, category or point records.

## Colab and measurement Actions

For measurements on your own Colab, use the [public Colab tool](tools/colab/README.md).
If Colab authentication has been configured, **Benchmark measurement** Actions
are also available. This measurement workflow reads a committed request JSON
specifying the Case, GPU and number of runs, then saves results in the
`benchmark-results` Artifact.

Extract `artifacts/benchmark.json` for each run from the Artifact and submit it
through the same issue form. This workflow does not register results in Neon
automatically. See the [requests guide](benchmarks/requests/README.md) for the
measurement request format. Contributors measuring directly on their own GPUs
or Colab can submit results without configuring Colab authentication on GitHub.

## Storing observations and adopting implementations

The database stores the original result, Plan, Case, observations, metrics,
submitter ID/username, issue information, receipt time, category, points and the
settings in effect at receipt. Independent measurements are appended as
additional observations. Submission outcomes are recorded as
`consistency_checked` / `self_reported`. This process does not rerun measurements
on a GPU and therefore does not guarantee that the measurements are authentic.

Database registration, kernel/Algorithm approval, leaderboard generation and
dispatcher adoption are separate processes. Aggregation, ranking and dispatch
generation from accumulated observations use [separate generation tools and
Actions](benchmarks/dispatch/README.md).

For central Actions/Neon configuration, database migrations, administrator
reruns and observation queries, see the [detailed contribution and operations
guide](docs/benchmark-contributions.ja.md).
