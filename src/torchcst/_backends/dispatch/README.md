# Plan selectors

**English** | [日本語](README.ja.md)

The common boundary for choosing a Plan from input conditions is
`Selector.select(context) -> DispatchDecision`. Represent full/window through
Registry Algorithm IDs, revisions and Recipes rather than special branches.
`LinearOptions` and `memory="full" | "window"` were removed, without a
compatibility loader.

## From data to execution

```text
Appended observations
  → Rank candidates with a scoring function
  → Obtain candidates and evidence for each condition
  → Generate/validate the selector and save its artifact
  → Construct the Selector when loading
  → Inputs/binding → Context → select(context) → Algorithm support validation → State preparation → execution
```

The scoring function may use speed, memory or other metrics. It is not called at
runtime. `score_policy` records its ID, revision and parameters as metadata; do
not embed the function in JSON or evaluate arbitrary code during forward.
[Generation tools](../../../../benchmarks/dispatch/README.md) provide fixed DB
snapshot extraction, scoring functions, leaderboards and exact-artifact
generation. Scoring functions run in those tools, not as Python execution inside
the DB, and do not automatically become the default policy.

| Selector | Current status | Selection method |
| --- | --- | --- |
| FixedSelector | Implemented | One explicit Plan for bootstrap/direct comparison |
| ExactSelector | Implemented | Exact-match table for observed conditions |
| OrderedSelector | Implemented | Check Algorithm routing conditions in explicit Plan order |
| Rules / decision trees | Future candidates | Return a Plan or no selection through the same `_match(context)` boundary |
| MLP and similar methods | Future candidates | Return a Plan or no selection through the same boundary |

The Selector ABC's `_match` proposes a candidate, and metadata-only `select`
delegates to the shared runtime. Dispatcher uses the same candidate
validation/fallback procedure. Check Algorithm support and workspace limits, and
consider the explicit fallback Plan only when unselected or unsupported. Do not
hide invalid Plans or implementation exceptions with fallback. No special
handling is based on Algorithm names.

Public execution uses Dispatcher.run(binding, inputs, plan=...). The binding
validates actual inputs, builds a Context and retrieves the corresponding State.
Direct execution with a manually constructed Context is not accepted. A forced
Plan bypasses selection and fallback, but not actual-input, support or State
validation. Registry owns only registration, Plan declaration validation and the
codec. Backward uses forward's Algorithm and saved state without reselection.
See [execution boundaries](../../../../docs/algorithm-dispatch-boundaries.md).

## Artifact v1

| Field | Meaning |
| --- | --- |
| schema_version | Version of the outer format and exact-match conditions; currently 1 |
| selector.kind / revision | Currently `exact_table`; selection-policy version |
| score_policy | Scoring function `id`, `revision` and `parameters` |
| dataset_snapshot | Identifier of the fixed dataset used for ranking |
| runtime_versions | Full PyTorch version and optional Triton version; checked at load time |
| plans | ID → Plan reconstructible and validatable by Registry |
| entries | List of `condition`, `plan_id` and `evidence_ids` |
| fallback_plan_id | Plan to validate for unobserved/unsupported conditions |

A condition stores the complete typed Operator declaration, including
Kernel/Chart/Geometry/Pattern and input/output layout types, settings and
revisions. It also distinguishes shape, stride, dtype, atom count, coordinate
width, GPU type/name/compute capability/SM count, required gradients, eager/Graph,
precision, determinism and workspace limit. Local GPU index is excluded. It does
not read atom Tensor values, broad/sharp initial-bandwidth labels or optimizer
state.

Declaration type names are strings to compare, not types to import or construct
from JSON. Reject duplicate conditions, unknown Plans/revisions, invalid recipes,
flags of the wrong type, nonfinite values and duplicate JSON keys. Explicitly
reject unknown kinds/schemas rather than implying that an MLP can be loaded.
Condition coverage, evidence trust and comparability of measurement conditions
such as GPU drivers are the responsibility of generation and approval. Loading
successfully does not certify performance.

## Usage

```python
from pathlib import Path
from torchcst import CSTLinear, presets
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import load_selector

selector = load_selector(Path("dispatch.json").read_bytes(), registry=REGISTRY)
layer = CSTLinear(
    chart=chart, atoms=p, kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT, selector=selector
)
```

The selector applies to all CSTLinear paths. CPU Torch references, CUDA and
Strip/Torus fused paths use the same Registry/Plan/Algorithm. Execution uses the
selector's own Registry; custom Registries are also supported. Keep the selector
as a separate artifact, outside mathematical declarations and state_dict. When
unspecified, the Module uses an explicit OrderedSelector. It checks routing
conditions for FULL, Torch normalized, factored and materialized in that order.
This is not automatic performance-based adoption.

FixedSelector and OrderedSelector accept operation-specific Contexts.
ExactSelector can also handle non-Linear Contexts. These serialize a frozen
dataclass Context as its type name and scalar/tuple/immutable-dataclass fields;
the condition is `{"context": typed_declaration}`. Tensors and mutable values are
rejected. The existing Linear artifact v1 condition format is preserved. Generic
Contexts compare every field; they do not implicitly inherit Linear's rule that
ignores GPU index. Type names remain comparison strings: loading performs no
imports or arbitrary code execution.

Generation can construct an artifact from ranked conditions and Plans:

```python
from torchcst._backends.dispatch import ExactEntry, ExactSelector

selector = ExactSelector.from_entries(
    [ExactEntry(context, winning_plan, (observation_id,))],
    registry=REGISTRY,
    fallback_plan=baseline_plan,
    revision="policy-v1",
    score_policy={"id": "my-score", "revision": "v1", "parameters": {}},
    dataset_snapshot=snapshot_id,
)
Path("dispatch.json").write_text(selector.dumps(), encoding="utf-8")
assert (
    load_selector(selector.dumps(), registry=REGISTRY).select(context).plan
    == winning_plan
)
```

JSON parsing and dictionary construction happen only at load time. Forward looks
up a dictionary using a metadata key. Fixed Operator keys are cached for up to
128 declarations, without retaining Tensors or gradient state. Runtime performs
neither scans proportional to the entry count nor DB/network/file I/O.
Dictionary lookup is average O(1); metadata-key generation and support validation
have separate costs. Recapture CUDA Graphs after replacing a selector. Existing
Graph replay uses the saved execution.

## Future selector changes

As observations and table size grow, first generate region partitions/rules.
Multiple performance modes alone are not a reason to switch automatically to an
MLP. Evaluate unseen shapes and separate sessions, comparing actual speed loss,
memory, inference time and artifact size before adoption. Consider an MLP or
similar model if rules cannot preserve the required performance.

Change the selector implementation and per-kind loaders while preserving
Context, Plan, Registry and execution boundaries. Retain exact-match tables and
raw observations as approximation references and evidence for unsupported
regions. Approximate selectors must also be able to return no selection;
extrapolation must not bypass mathematical contracts or workspace validation.
Distinguish measured memory peaks from Algorithm workspace upper bounds.

## Validation

```bash
python -m pytest -q tests/test_dispatch_selectors.py tests/test_cuda_dispatch.py tests/test_linear_parameters.py
python -m pytest -q
```

The local CPU suite on 2026-10-02 recorded 602 passed / 133 skipped. Checks cover
JSON round trips, every condition field, unobserved conditions, explicit fallback,
workspace limits, different Algorithm IDs, custom rules, two forwards with
autograd, and checkpoints used with a replaced selector. CUDA and PostgreSQL
tests were skipped because those environments were not connected. L4 validation
covered the public path loading an actual file, independent oracles, Graph
replay, and complete-step time/peaks. The four target files recorded
108 passed / 0 failed / 0 skipped. See the
[GPU validation record](../../../../docs/dispatch-json-verification.ja.md).
