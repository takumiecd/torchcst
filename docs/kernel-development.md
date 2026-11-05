# Developing and improving kernels

**English** | [日本語](kernel-development.ja.md)

See the [README](../README.md) for the current public API. The old local-product
prototype has been removed. Building on the development policy established on
2026-10-05, the examples below use the existing Plans and Cases. Research on small
CST linear transforms uses the existing Linear runner, PostgreSQL storage and
dispatch generation. Integrate into main through GitHub PRs.

For the development entry point and PR workflow, see
[CONTRIBUTING.md](../CONTRIBUTING.md). The owner's GPU and research-worktree rules
are in the [research operating rules](research-operations.md). This guide covers
implementation, registration and validation details.

When implementing or optimizing profile-product Linear on RegularGrid, see
[kernel design around H generation, reuse and lifetime](h-lifecycle-kernel-design.md).
Treat shared X→H reads, local H→Y aggregation, and retention/recomputation through
backward as one execution plan. Explain each candidate's unit of work, retained
storage and complete-step time in those terms.

## Types of additions and files to change

A CUDA kernel here is a computational implementation. Adding a mathematical
KernelSpec also requires changes to declarations, the Torch reference
implementation and the operation's contract.

| Change | Implementation and declarations | Benchmarks and validation |
| --- | --- | --- |
| Recipe/launch tuning for an existing Algorithm | Its `recipe.py`, `validate_recipe` and executor/kernels. Local research routes live in `benchmarks/cuda/linear/local_product.py` | New Plan and comparison Case, JSON round trip, independent oracle under the same contract and GPU measurements |
| New CUDA Algorithm with the same mathematical contract | Algorithm/Recipe, required contracts and executor/kernels in `algorithms/<method>/` | Register in the benchmark Registry; add catalog/Case, oracle, supported/unsupported conditions, autograd and Graph validation |
| New mathematical Kernel/operation | Declarations and Torch reference implementation in `kernels/`, required Operator/Chart contracts and supporting Algorithms | Add corresponding fixtures, an independent oracle, protocol, result adapter and submission/dispatch validation |

Put Linear implementations in each backend's `algorithms/linear/`. Atom
coordinate updates are a separate operation, `atom_update`: its Torch reference
is in `algorithms/atom_update/`, and concrete Polar update implementations belong
in `algorithms/polar_update/`. Do not add a per-Kernel dispatcher to the optimizer.
All updates use the same AtomUpdateInputs/Context and common Dispatcher; general
updates, including Sphere/Torus, fall back to the Torch reference. Treat the
Linear Plan, update Plan and base optimizer separately. The existing runner's
`--polar-update` selects only an update candidate; it does not change the Linear
Plan.

Candidates with the same mathematical meaning retain the same semantics.
Changes to normalization, support or differentiation require a distinct
semantics version. Do not register duplicate Plans by changing only an alias.
When changing the meaning of a registered recipe, update the implementation
contract revision and state which comparisons with older results remain valid.

### Implementation location and interface for a new Algorithm

The smallest working examples are
[`window/algorithm.py`](../src/torchcst/_backends/cuda/algorithms/linear/normalized_euclidean_strip/window/algorithm.py)
and [`window/recipe.py`](../src/torchcst/_backends/cuda/algorithms/linear/normalized_euclidean_strip/window/recipe.py).
The common interface is defined by
[`Algorithm`](../src/torchcst/_backends/algorithm.py).

| Element | What to implement |
| --- | --- |
| `id` / `revision` | An implementation ID and contract version that distinguish it from other candidates |
| `operation_id` / `semantics_id` | The supported operation and mathematical semantics |
| `recipe_type` / `validate_recipe` | An immutable dataclass and strict validation of accepted settings |
| `supports(context, recipe)` | Compatibility with Kernel/Chart contracts, shape, stride, dtype, GPU, precision and gradients; return reasons for unsupported conditions |
| `workspace_bound` | An upper bound on implementation-managed scratch, or `None` if unknown; distinguish it from a measured GPU peak |
| `input_type` | Typed Inputs shared by the operation; the input type defines required and optional fields |
| `execute(state, inputs)` | Lazily import the executor from the binding/Recipe and actual inputs, and connect backward to the corresponding saved forward state |

Algorithm construction, recipe validation and support checks must not import
GPU code or read Tensor values. Do not retain inputs, support or gradient
buffers on an Algorithm instance. Each autograd invocation owns its saved
forward state; the optimizer updates Parameters. `supports` uses metadata only.
Support classification that depends on the current sigma is updated at execution
time.

### Connecting the Registry, runner and adapter

First register a new research Algorithm in the benchmark-local `REGISTRY` in
[`benchmarks/cuda/linear/manifest.py`](../benchmarks/cuda/linear/manifest.py).
Check that importing metadata does not load Triton. Connecting it to the
production Registry or the public CSTLinear selector is a separate decision
after hardware validation.

This example registers the existing normalized Window in an independent Registry
and checks a JSON round trip:

```bash
python - <<'PY'
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.window.algorithm import NormalizedWindowAlgorithm
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.window.recipe import WindowRecipe

registry = Registry()
algorithm = NormalizedWindowAlgorithm()
registry.register(algorithm)
plan = ExecutionPlan(algorithm.id, algorithm.revision, WindowRecipe())
assert registry.loads_plan(registry.dumps_plan(plan)) == plan
print(registry.dumps_plan(plan))
PY
```

Check your Algorithm/Recipe in the same way, and add tests for rejected settings,
unknown revisions and unsupported contexts. Normalized full/window permit only
their registered default recipes; changing numeric values in the JSON alone
does not create an executable new candidate.

Even for a candidate with the same mathematical contract, check every path:
runner fixture construction, Plan execution, source-hash collection, and the
correctness/measurement workers. The existing Linear runner supports normalized
Strip and small local polar-product fixtures; it does not automatically support
arbitrary new operations. When adding a fixture or result metric, update
[`fixtures.py`](../benchmarks/cuda/linear/fixtures.py),
[`protocol.py`](../benchmarks/cuda/linear/protocol.py),
[`run.py`](../benchmarks/cuda/linear/run.py),
[`database/adapters/linear.py`](../benchmarks/database/adapters/linear.py), and the
submission and dispatch decoders together. Do not force it into an existing
fixture's contract. Provide an explicit adapter/ingestion entry point for a new
result format. If a metric's meaning changes, increment the adapter revision and
retain past projections.

## Try the workflow with a registered Algorithm

This example uses the implemented normalized full as the baseline and window512
as the candidate to demonstrate registration, comparison setup, validation,
measurement and checks before submission. It does not produce performance
results for a new kernel. The first two commands require only a CPU.

```bash
python -m tools.kernel_dev prepare \
  --case benchmarks/cuda/linear/cases/normalized-1024-broad.json \
  --candidate window512 --output output/window-comparison
python -m tools.kernel_dev check \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json
```

The output consists of `plans.json` containing only the selected
baseline/candidates, `case.json` retaining the same inputs, optimizer and dense
reference, and `source.json` containing the original Case/catalog paths and
SHA256 hashes. Use a new output directory. Repeat `--candidate` for multiple
candidates. After registering a new Plan with the same mathematical contract in
the catalog, you can select its alias as a candidate. The following GPU
validation checks execution-time compatibility and numerical correctness.

On a supported CUDA GPU, run from the repository root:

```bash
python -m tools.kernel_dev test --suite normalized-strip
python -m benchmarks.cuda.linear.run \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json \
  --source-commit "$(git rev-parse HEAD)" --correctness-only \
  --output output/window-comparison/correctness.json
python -m benchmarks.cuda.linear.run \
  --plans output/window-comparison/plans.json \
  --case output/window-comparison/case.json \
  --source-commit "$(git rev-parse HEAD)" \
  --output output/window-comparison/benchmark.json
python -m benchmarks.submissions check output/window-comparison/benchmark.json
```

If correctness fails, fix it before measuring. The measurement runner also runs
correctness checks first. Commit a new implementation before measuring it, and
retain the actual source hashes in the results. Follow
[CONTRIBUTING.md](../CONTRIBUTING.md) to submit completed result JSON and include
evidence in the PR.

## Development environment and validation commands

Run all commands from the repository root. `tools.kernel_dev` is a development
entry point within the repository and is not included in the distributed wheel.
It does not allocate GPUs, store results in the database or update dispatch.

```bash
python -m pip install -e '.[dev]'
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-128-initial.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-compact-keyed-initial.json
python -m tools.kernel_dev test --suite cpu
```

`check` uses the existing runner's manifest/Registry to validate registration and
JSON round trips for all Plans; candidates, baselines and mathematical contracts
for each Case; and worker snapshot round trips. Results include SHA256 hashes of
the input files. It imports no GPU implementation and requires no GPU or
database. Repeat `--case` to check multiple Cases; omit it to check only the
catalog. PASS here establishes declaration consistency, not gradient correctness
or performance.

`test --suite cpu` runs all pytest tests in an environment without CUDA. On a GPU
host, use `CUDA_VISIBLE_DEVICES='' python -m tools.kernel_dev test --suite cpu`.
GPU and live-database tests are skipped when their requirements are absent, so
record pytest skip counts as well.

In a GPU development environment, run
`python -m pip install -e '.[dev,cuda]'`, then use
`python -m tools.kernel_dev test --suite atom-update` or
`python -m tools.kernel_dev test --suite normalized-strip` as appropriate.
Validate Atom updates with `python -m tools.kernel_dev test --suite atom-update`.
These suites fail without CUDA/Triton rather than treating GPU validation as
complete through skips. They provide entry-point regression coverage; choose
additional boundary, gradient and layout tests for the change. Continue using
the existing Linear runner for performance measurements.

## Source and comparison conditions

1. Create a named `kernel/<purpose>` research branch and worktree from main.
   Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for branch naming when adding or
   optimizing kernels or tuning recipes/launches.
2. Put implementations in `src/torchcst/_backends/cuda/algorithms/`. Separate
   mathematical specifications from execution settings. Local product belongs
   in `linear/local_product/`; research Plan registration belongs in
   `benchmarks/cuda/linear/local_product.py`. Preserve existing Plan settings and
   semantics, and give each new candidate an explicit route.
3. Add candidates to `benchmarks/cuda/linear/plans-*.json`, and explicitly identify
   the comparator, baseline and dense reference in `cases/`. Do not introduce a
   fixed σ; distinguish initial bandwidth from bandwidth updates at each step.
4. Check declarations with the existing runner's `--validate-only`. Compare time
   and memory using the same initial Parameters, inputs, dtype and optimizer
   contract.

Use `tools.kernel_dev check` to check multiple Cases together. After adding a
route, check that its declarations can be restored from the registry and that
the Case containing the candidate explicitly identifies baseline/dense. Record
comparisons that change bandwidth or shape as separate Cases.

Do not embed a new implementation in the benchmark directory or add another
runner. The existing `research_local_product` is a research registry, not the
public CSTLinear default.

## Correctness and GPU measurements

Use an independent FP64 scalar oracle to validate Y, dX and all atom gradients.
Cover normalization, support boundaries, a single site, empty support, a cropped
domain and retained backward. When changing updates, compare Parameters, AdamW
moments and step counters against the public CSTOptimizer. Check variable σ,
moving support/layout and Graph replay.

Submit L4 work through the [shared pool](../tools/colab-l4-pool/SKILL.md). Before
submission, inspect `git status` and ensure raw evidence does not enter the
snapshot. Put a separate driver in ignored `evidence/` and have it call the
existing runner and validation. A driver is not a new measurement runner.

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-compact-keyed-initial.json \
  --phase-diagnostics --kernel-diagnostics \
  --output benchmarks/cuda/linear/evidence/comparison.json
```

Use complete training steps without instrumentation for adoption timing. Measure
diagnostics in a separate Graph; do not add their times together or subtract
them from the primary measurement. Distinguish peak allocated and reserved
memory, including capture/replay, and mark GPU process usage as unmeasured unless
you measured it. Do not count 21 timing samples as 21 independent runs. Remeasure
small improvements in an independent job. Retain negative results too.

## Database storage and dispatch candidates

Append complete runner JSON through the existing
[database entry point](../benchmarks/database/README.md). Check that exporting
reproduces the original bytes and that resubmission with the same provenance
works. Include the owned-pool job ID, source snapshot SHA256 and result archive
SHA256; do not transfer credentials to the GPU. The research local-product
adapter revision is 3; normalized Strip uses 2.

Use a [generation request JSON](../benchmarks/dispatch/README.md) to specify the
source, GPU/runtime, Case, timing objective, memory policy and minimum number of
independent runs. Generate the dataset, leaderboard and dispatch JSON, and check
fallback behavior for unknown conditions. Database storage, candidate generation
and public default adoption are separate operations.

The current local-product execution-condition key does not include the initial
rho distribution. Automatic selection across Cases with the same shape but
different bandwidth distributions is not complete, so generate candidates per
Case. `baseline-peak` is a comparison policy, not a user-specified memory limit.

## Records and integration

[CONTRIBUTING.md](../CONTRIBUTING.md) is authoritative for development,
submission and PR validation requirements. The
[research operating rules](research-operations.md) are authoritative for the
owner's shared-GPU restrictions and research-worktree evidence preservation.
Keep concise local-product records in `docs/research-history/local-product/` and
CUDA Linear records in `docs/research-history/cuda-linear/`. Record the decision,
errors, complete-step time and peaks, independent run count, job ID,
source/result hashes, reproduction commands and preservation locations. Keep
candidates on research branches until hardware validation is complete.
