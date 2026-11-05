# Algorithm execution boundaries

**English** | [日本語](algorithm-dispatch-boundaries.ja.md)

This contract implements the boundary policy established on 2026-10-05. See the
[README](../README.md) for the public API and [Atom and Algorithm state
ownership](atom-state.md) for ownership. Dispatcher and ExecutionBinding are
backend-wide interfaces; LinearInputs and LinearBinding belong to the operation.
Linear and atom-coordinate updates have separate Inputs/Context types and use the
same Registry and Dispatcher.

## Selecting a Plan or executing one directly

Use `model` and `x` from the [README training
example](../README.md#training-with-a-standard-optimizer). This example selects a
Torch materialized Plan without changing the mathematical declaration:

```python
from torchcst import (
    DefaultRecipe, Dispatcher, ExecutionPlan, FixedSelector,
    LinearBinding, LinearInputs, make_algorithm_registry,
)

registry = make_algorithm_registry()
plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
model.selector = FixedSelector(plan, registry=registry)
y = model(x)

# Refer to the same live Operator; do not copy Parameters.
binding = LinearBinding(model.operator)
flat = x.reshape(-1, model.in_features).contiguous()
y_direct = Dispatcher(registry=registry).run(binding, LinearInputs(flat), plan=plan)
torch.testing.assert_close(y, y_direct)
```

`backend=` is an alias for built-in Plans; an explicit `selector=` takes
precedence. The selector's registry serves both Torch and CUDA. The default
ordering retains supported FULL execution, the CPU normalized reference, and the
size policy for factored/materialized execution. An explicit Plan still undergoes
input, precision and workspace support checks.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Inputs | Actual data for one call; the operation's required/optional input and output contract |
| Caller providing a binding | References to operation settings and live owned state; input validation, Context construction and AlgorithmState retrieval |
| Context | Immutable execution-condition metadata needed for selection and support checks |
| Registry | Algorithm registration/retrieval, Plan/Recipe declaration validation and Plan JSON serialization |
| Selector | Policy for proposing a Plan from a Context; may return no selection |
| Dispatcher | Ordering of validation, selection, state retrieval and execution; fallback before execution |
| Algorithm | Declaration of accepted input contracts, support checks, and creation/preparation/execution of dedicated state |
| AlgorithmState | Reference to its binding; the Algorithm's layout, reusable buffers and prepared generation |
| AtomState / AtomOptimizerState | Authoritative Atoms, ID-to-row mapping, layout generation and training state that follows the layout |

Do not add branches for Linear, Polar or optimizers to shared code. Keep concrete
preparation, layout and computation inside each Algorithm's directory. Sharing an
input type does not require a family-specific Algorithm inheritance hierarchy:
an input type defines the operation's call contract; an Algorithm defines a way
to compute it.

## Per-call inputs, settings and state

Separate actual data that changes on each call into Inputs, bound operation
settings into the binding, and retained memory into State. For Linear, Inputs
contains `x`, the binding references the existing Operator, and State references
AtomState and owns Algorithm-specific layouts/buffers. Obtain Chart, Geometry and
Kernel from the existing Operator; do not add a normalized-only Operator or adapter.

Do not independently combine `x, parameters, operator, site, state` at the
execution entrypoint. Obtain authoritative Parameters from the binding's
AtomState and the live Chart/Kernel from its Operator. Distinguish declaration
snapshots from actual Tensor references, and do not replace trainable Chart values
with a declaration snapshot. Contiguous copies, packing and decoding are the
Algorithm's work, not a second authoritative copy of training state.

Define required inputs as fields without defaults and optional inputs as fields
with explicit defaults. If omission differs from `None`, represent that distinction
in the input type; the shared Dispatcher must not fill values implicitly. Reject
unknown fields, invalid types/shapes and missing required inputs before selection.
Do not duplicate required/optional field lists in the Registry or give every
operation an optional `x`.

Candidates for the same operation use the same Inputs and output/update contract.
Each Algorithm declares its accepted input type and checks its own constraints in
addition to common operation-level validation. Packing and support tables needed
only by a fast path must not become required user inputs; the Algorithm prepares
them. AtomUpdateInputs(previous, step_size) requires a pre-update snapshot and the
outer step size, and obtains the optimizer proposal from the current Parameters
referenced by the binding. Before selection, validate the snapshot's shape,
dtype, device, non-aliasing and detached status, and require a positive step size.

## Bindings and owners

The Algorithm has not been selected when the selection Context is constructed.
Context construction therefore cannot depend on the selected AlgorithmState.

A binding is an interface provided by a Module or direct-execution caller, not a
new Parameter owner. At minimum, it must:

- Define the input type and operation contract, and perform common Inputs validation.
- Construct a Context from Inputs and live owned state.
- Retrieve the AlgorithmState for the selected Algorithm/Recipe.

Public Modules provide this interface and continue to own AtomState and the
AlgorithmState cache. Dispatcher, Selector and Registry retain no Parameters,
moments or per-owner State caches. Shared code must not require the concrete
CSTModule type. Direct Plan callers provide the same interface and may use Atoms
outside a Module. Add concrete binding classes only where an operation needs
them; do not require a wrapper class for every operation.

## Context

Context describes the current execution conditions derived from Inputs and the
binding. It retains no Tensors, Parameter values, moments or mutable State
objects. It does not read input contents back to the host, decode values or run
kernels. Inputs/bindings validate required inputs; a Context may validate only its
own metadata types, ranges and consistency.

The shared boundary contains only `operation_id` and `workspace_limit_bytes`.
Other metadata is operation-specific. LinearContext describes the Operator
declaration, input shape/stride, dtype, device, atom count, coordinate width,
required gradients, precision, determinism and eager/Graph settings.
AtomUpdateContext describes the Kernel, center Geometry, Parameter layout and
scalar placement. It requires neither Linear input shapes nor Linear gradient
requirements, and its execution `operation_id` is `atom_update`.

## Atom updates from an optimizer

CSTOptimizer sends atom updates for every site to the shared Dispatcher without
branching on Kernel type. The default is `torch_atom_update`. Torch and fused CUDA
Polar implementations are candidates for the same `atom_update` operation, chosen
by `update_selector`. Linear Plans and update Plans are independent. Selection
and support checks also happen before the base optimizer updates, so unsupported
or invalid Plans do not advance Parameters or moments.

The Sphere/Torus Torch fallback uses Geometry retraction. Preserve the existing
gradient projection and vector-state transport through OptimizerStateAdapter; do
not bypass them by directly accepting the Torch optimizer's proposal. Preserve
the existing optimizer checkpoint format and AtomState ownership. Registering a
fused Polar candidate does not make it the default. Graph support is specific to
each Algorithm; this contract does not add capture support to CSTOptimizer as a
whole.

In public execution, users do not construct a Context by hand: the Dispatcher
constructs it through the binding. Metadata-only Contexts may be constructed for
declaration checks and candidate comparisons, but are not proof that actual data
has been validated. Direct Plan execution also reconstructs the Context from
Inputs/bindings. Update declaration metadata and obtain required device
information before CUDA Graph capture.

Matching Contexts do not imply matching Parameter values or decoded results.
Layout versions also do not guarantee freshness of numerical caches. The
Algorithm refreshes atom-value-dependent decoding and support preparation on
every execution.

## From selection to execution

Public execution proceeds in this order:

```text
dispatcher.run(binding, inputs)
  1. The binding performs common Inputs validation
  2. Construct a Context from the binding
  3. The Selector proposes a candidate Plan
  4. Validate ID/revision/Recipe in the Registry and retrieve the Algorithm
  5. Check the Algorithm's support conditions and workspace limit
  6. If unsupported/unselected, perform 4 and 5 for the explicit fallback candidate
  7. Retrieve the selected Algorithm/Recipe's State from the binding
  8. The Algorithm checks State/actual-input consistency and prepares as needed
  9. Execute Algorithm.run(state, inputs)
```

For a fixed Plan or a strict FixedSelector, an explicitly supporting binding may
retain one selected-metadata entry and AlgorithmState reference, reusing the
support/workspace decision under the same execution conditions. The Dispatcher
itself retains no binding, Tensor or State. Inputs validation, live Context
construction, and Plan declaration type/schema/Recipe validation still run on
every call. Changes to settings, placement, owner, layout generation,
shape/stride, required gradients or precision settings invalidate the cache;
custom selectors select on every call. Atom numerical updates do not invalidate
it: value-dependent classification and computation run on the GPU every time.
The cache is excluded from checkpoints and deepcopy.

Optimizer preflight can be shared through `Dispatcher.prepare`, but after base
optimizer hooks run, conditions must still be checked using fresh Inputs/Context.
Prior validation is not proof that current inputs have been validated.

Use one shared candidate-validation procedure rather than separate rules in the
Selector and Dispatcher. The Selector owns policies such as ranking, search or
exact-match tables. Distinguish no selection/unsupported candidates from invalid
Plans; fallback must not hide unknown IDs, revisions or broken Recipes. Execute
only supported candidates and return the operation's result to ordinary callers.
DispatchDecision can expose the selected Plan and reasoning for diagnostics, but
must not store Tensors or State.

For forced Plan execution, pass `plan=` to the same Dispatcher entrypoint. This
bypasses only the Selector, not input validation, support checks or state
preparation; unsupported Plans raise an error. Do not create separate execution
paths through Registry.execute and Dispatcher.run.

## Fallback and state preparation

Fallback applies when valid Inputs are unsupported by a candidate or the
Selector makes no selection. Specify a Torch Algorithm with the same operation
semantics, normalization, required gradients, outputs and update contract. If
Torch is also unsupported, reject with a reason. Matching operation names alone
must not permit fallback to different semantics.

Omitted optional inputs follow the input type's default rules, independently of
fallback. Do not catch exceptions after state preparation or computation starts,
OOMs or failed optimizer updates and rerun through another Algorithm. Do not hide
partial updates or advance an optimizer step twice. If selection needs an
Algorithm-specific actual-input constraint, express it through metadata available
in the Context beforehand.

Retained-state types are also part of an Algorithm's contract. Atom-based methods
reference AtomState through the shared AlgorithmState, but not every operation
requires Atoms or an Operator. Methods without retained memory may use their own
empty State. The shared Registry must not switch execution paths using `state=None`.

AlgorithmState reuse keys distinguish Algorithm identity/version, Recipe,
operation binding and owner identity. Each Algorithm checks preparation-dependent
conditions such as input shape/device/dtype and rebuilds its dedicated
layout/buffers when they change. The default layout-generation check alone does
not establish that preparation is current. Recipe or operation-setting changes
require a different State or explicit invalidation.

Distinguish retained memory from temporary memory, and state which
preparation/execution allocations the workspace upper bound covers. Checking that
bound does not guarantee a measured GPU-wide peak. Other AlgorithmState caches
retained by the owner have separate costs. An unknown upper bound does not
satisfy a specified limit.

## Layout, backward and Graph execution

Algorithm-specific permutations and copies stay inside AlgorithmState and do not
advance the authoritative layout version. To move authoritative Atoms, use
AtomState.relayout at an explicit eager step boundary. Do not move them as a side
effect of Dispatcher selection, ordinary preparation or forward execution. Move
Parameters, gradients and declared optimizer state using the same mapping, then
reprepare stale State on the next run.

Backward uses the Algorithm chosen for forward and that call's saved state; it
does not redispatch. Do not overwrite saved values for pending backward calls in
reusable buffers. Preserve AtomState lifetime protection. CUDA Graph replay uses
the captured execution; changes to the Selector or binding require recapture.
This contract does not newly guarantee Graph support for optimizers or relayout.

## Code layout and migration

Shared implementation lives in `_backends/dispatch/`. Registry, Algorithm and
AlgorithmState use the existing shared files. Concrete Inputs/binding behavior
belongs to the code defining the operation contract; Algorithm-specific layouts,
kernels and actual-input constraints stay inside each Algorithm. Linear callers
use the existing Module/Operator. The shared Dispatcher does not depend on
concrete types in operators or nn. Do not change operation contracts to suit a
backend.

The migrated boundaries are:

| Before | Current |
| --- | --- |
| Context.validate_inputs validated Tensors | Inputs/bindings perform common validation; Context contains metadata only |
| Registry.validate also checked support/workspace | Shared dispatch code queries the Algorithm |
| Registry.execute joined validation and execution | Dispatcher is the sole public execution path |
| Selector.select also owned Registry validation and fallback | Selection policy is separate from shared candidate validation/fallback |
| Module selected Plans and assembled named inputs | Module provides a binding and passes Inputs to Dispatcher |
| create_state(atom_state, **configuration) | Construct references and dedicated State from the binding/Recipe |

The public README was updated while retaining existing Plan/selector artifact
formats. Registry.validate/execute and Context.validate_inputs were removed,
without compatibility wrappers. Selector.select is a metadata-only convenience
API using the same candidate validation/fallback procedure as Dispatcher. Forced
Plans use the `plan` argument of Dispatcher.run.

Regression checks cover required/optional/unknown inputs, agreement between
Context and actual data, custom Registries, unrelated operation inputs, Torch
fallback, forced Plans, invalid Plan rejection, state reuse/invalidation, shared
Atoms/moments, backward/Graph lifetimes, and declaration imports that do not load
computation code. In addition to CPU regressions, compare GPU oracles, complete
steps and peaks.
