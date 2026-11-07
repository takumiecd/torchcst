# Declaration and execution layout

**English** | [日本語](README.ja.md)

Separate types that declare mathematical meaning from implementations that
compute it on Tensors. Declarations contain no backend selection,
forward/backward implementation or GPU launch configuration. Backends preserve
the declared semantics and execute through supported Algorithms and Recipes.

The shared Algorithm execution/state lifecycle lives in `_backends/algorithm.py`
and `state.py`. AtomState in `atoms/state.py` is the authoritative Module-owned
training state. `atoms/optimizer_state.py` holds optimizer state and declarations
for following layout changes; `optim/binding.py` connects it to Torch. Concrete
computation, layouts and buffers belong inside each Algorithm. See the
[state ownership and relayout contract](../../../docs/atom-state.md) and the
[Algorithm execution contract](../../../docs/algorithm-dispatch-boundaries.md)
for Inputs, Context, Registry and Dispatcher boundaries. The layout below
describes the current implementation.

## Current layout

```text
torchcst/
  geometry/
    spec.py / presets.py       Immutable space/metric declarations and construction
    state.py                   GeometryState fixed metadata and checkpoints
  patterns/
    spec.py / presets.py       Line / Grid / Points axis declarations and construction
    state.py                   PatternState bounded Tensor state
  charts/
    base.py                    ChartSpec / ChartState ABCs
    explicit.py                ExplicitChartSpec / ExplicitChartState
    product.py                 ProductChartSpec / ProductChartState
    regular_grid.py            RegularGridChartSpec / RegularGridChartState
    presets.py / __init__.py    Pure declaration construction / compile_chart
  kernels/
    profiles/                  Function-shape declarations such as Triweight / Gaussian
    parameterizations/         Amplitude / bandwidth / activity coordinate interpretation
    normalization.py           Normalization rules, domains and floors
    spec.py                    ProfileBinding / CoordinateSpec / KernelSpec / state policy
    factories.py               Pure composition of Profile / Factor / CoordinateSpec
    options.py                 Execution settings separate from mathematical Specs
    state.py                   Shared fixed Tensor state and checkpoints
  operators/
    spec.py                    Single Chart / Chart pair / linear Operator declarations
    binding.py                 Operator referencing Module live state
    context.py / execution.py  Linear metadata / typed Inputs / direct-execution binding
    atom_update.py             Shared atom-update Inputs / Context / live binding
    metadata.py                Declaration cache shared by operation bindings
  _backends/
    algorithm.py / state.py    Backend/operation-wide Algorithm and per-owner execution state
    schema.py                  Context protocol / Plan / support / decision
    registry.py                Shared registration and Plan declaration/JSON validation
    dispatch/                  Runtime / candidate validation / replaceable selectors
    serialization.py           Strict JSON codec for declaration metadata
    catalog.py                 Lazy registration of built-in Torch/CUDA Algorithms
    cuda/
      _host_function_metadata.py / .cpp
                              CPU callback/storage guard shared by Linear / Polar
      algorithms/
        linear/
          normalized_euclidean_strip/
            _shared/           Computation actually shared by full / window
            full/              algorithm / recipe / executor / kernels
            window/            algorithm / recipe / executor / provider / kernels
          radius_cell_mass/   Native radius-cell integration / preparation / autograd
        polar_update/         CUDA atom-update optimization candidate / executor / kernel
    torch/
      algorithms/
        linear/               materialized / factored / normalized_radial
        atom_update/          Reference coordinate-update Algorithm for all Kernels/Geometries
        polar_update/         Polar-update Algorithm / coordinate law / Graph execution path
      geometry/                Embedding / distance / projection / update / transport
      patterns/                Grid expansion / point selection / bounds
      charts/                  Chart coordinates / subregions / support
      profiles/                Profile evaluation / derivatives
      parameterizations/       Actual amplitude / bandwidth / activity interpretation
      kernels/                 Atom composition / weights / initialization / factor derivatives
      updates/                 Kernel-coordinate updates and optimizer-state transport
                               Polar laws live in polar_update/executor.py
      operators/               Torch linear execution receiving an Operator
  nn/
    Public Modules             Parameter / buffer / checkpoint ownership
```

The runtime must not depend on `benchmarks/` or `tests/`. The Torch backend is a
reference path using Torch Tensor operations on supported devices, not a CPU-only
backend. Do not mix NVIDIA-specific Triton optimizations into it.

## Declaration categories

| Declaration | Fixed semantics | Execution settings and implementation |
| --- | --- | --- |
| Geometry | Space, dimension, metric, radius and stored center representation | Coordinate conversion, distance, projection and update procedures |
| Chart / Pattern | Shape, point placement, flatten order, origin, spacing and physical Strip pitch | Coordinate expansion and GPU layout/routing procedures |
| Profile | Function shape over dimensionless distance | Evaluation and derivative implementation |
| Parameterization | Rules/derivatives interpreting atom coordinates as amplitude, bandwidth and center; semantic optimizer-field blocks | Interpretation computation and Tensor packing |
| Normalization | Domain, rules such as L2, and norm floors | Reductions, saving/recomputation and workspace |
| Kernel | Composition of radial / separable / amplitude factors | materialize / full / window / fused methods |
| Initialization/update policy | Meaning of initial distributions and coordinate updates | Sampling, projection, retraction and state-transport computation |

Sphere / FlatTorus retain the current chord metric. Recipes must not change
normalization domains or coupled amplitude-bandwidth derivatives. A change to
the numerical contract requires a distinct semantics revision and must not be
mixed with existing checkpoints or fallback for the same operation.

## Where shared computation belongs

Preparation/computation used by one Algorithm belongs inside that Algorithm.
Code actually shared by multiple Algorithms for the same operation belongs in
that operation group's `_shared/`. Share CUDA-specific code across operations
inside CUDA only when there is a demonstrated need. Do not preemptively create a
backend-wide `shared/` for possible future MPS/Radeon reuse. Shared mathematical
meaning belongs in declarations and is distinct from sharing implementations.

## Current connections

CSTLinear / CSTConv2d construct concrete ChartState and shared KernelState from
ChartSpec / KernelSpec. Geometry, Pattern, Kernel and Profile use shared State
types; Chart State is split into Explicit / Product / RegularGrid. There are no
compatibility methods or old checkpoint loaders. Operator references the
ChartState / KernelState / Atoms owned by the Module. Layout types distinguish a
single Chart from an input/output Chart pair. Torch evaluation and updates run in
backend functions, using current Tensors/buffers rather than rebuilding
declaration snapshots on every forward. See [Operator](../operators/README.md).

Registry depends on neither a backend nor an operation. It handles only
registration, retrieval and Plan declaration validation. The shared
Dispatcher.run(binding, inputs, plan=...) connects input validation, Context
construction, selection/support checks, State retrieval and Algorithm.run. Forced
Plans use the same path. The Context protocol contains only operation_id and
workspace_limit_bytes, without Tensor validation or training state.
Selector.select's metadata validation uses the same procedures in
dispatch/validation.py and runtime.py. Registry.validate/execute were removed;
there are no execution compatibility wrappers.

CSTLinear provides LinearInputs(x) and the binding interface and continues to own
AlgorithmState. LinearBinding connects an existing Operator, or declarations and
Parameters, outside a Module. Algorithm.input_type declares concrete operation
inputs; Polar and optimizers do not require Linear inputs. Algorithm creates
State from a binding/Recipe and computes through execute(state, inputs). State
for operations without Atoms uses the same path. CSTLinear backend names are
aliases for existing Plans; execution uses the selector's own Registry.
Registration does not load computation code.

CSTOptimizer creates an AtomUpdateBinding for every site and sends
AtomUpdateInputs(previous, step_size) to the shared Dispatcher. All candidates
use operation_id atom_update. The optimizer contains no Polar type branch or
dedicated dispatcher. The default, torch_atom_update, applies Kernel coordinate
laws and Geometry retraction. update_selector is replaceable independently of
the Linear selector. torch_polar_update and cuda_polar_update_fused are candidates
for the same operation; support conditions belong to the Algorithm. Existing
Torch reference gradient projection and OptimizerStateAdapter vector transport
are preserved. The base optimizer computes AdamW and other proposals, moment
updates and step counts; AtomState owns the state.

The existing Operator is the sole entrypoint for obtaining Chart/Geometry/Kernel;
there is no normalized-only Operator. Extraction and support checks for regular
site layouts live in the Torch normalized Algorithm's `layout.py`. CUDA
FULL/WINDOW reuse this pure metadata processing while retaining their own
execution, preparation and kernels. Modules refresh declarations using buffer
versions; numerical atom preparation uses current values on every call.

Operation contracts live outside backends. Distinguish Algorithm/Recipe settings
from kernel technologies; do not collect different computation methods in one
file simply because they use Triton. Do not create unimplemented directories or
empty templates.

The radius-cell-integral Profile is declared as ChebyshevSpec in
`kernels/profiles/fourier.py`, and its Factor as RadiusIntegratedProfileBinding.
CoordinateSpec independently declares initialization/update laws;
`factories.product` composes them into the existing KernelSpec. Torch coefficient
generation lives in `_backends/torch/profiles/radius_fourier.py`; CUDA
implementation lives in `_backends/cuda/algorithms/linear/radius_cell_mass/`.
`product` multiplies factors per Chart; `radial` evaluates once after summing
Chart distances. The Torch radial reference for two Charts lives in
`kernels/joint_radial.py`. Distinguish per-Factor normalization from normalization
of the complete operator.

Tests belong in `tests/`; ongoing correctness, time and memory measurements
belong in `benchmarks/`. Keep unadopted methods on separate research branches,
without prototype directories in the runtime tree. Preserve past decisions in
`docs/research-history/`. Registration, support checks, approval and adoption by
default dispatch are separate stages.

See [Kernel](../kernels/README.md), [Geometry](../geometry/README.md),
[Pattern](../patterns/README.md) and [Chart](../charts/README.md) for details.

## Plan JSON export/import

ExecutionPlan declares execution settings. The registry reconstructs Plans using
the registered Algorithm's recipe type and validates the schema, ID, revision,
all recipe fields and their values.

| Registry API | Conversion |
| --- | --- |
| `dump_plan(plan)` / `load_plan(data)` | Plan ↔ JSON-compatible dict |
| `dumps_plan(plan)` / `loads_plan(text)` | Plan ↔ JSON text; UTF-8 bytes can also be loaded |

```python
from pathlib import Path
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.plans import WINDOW

path = Path("plan.json")
path.write_text(REGISTRY.dumps_plan(WINDOW), encoding="utf-8")
restored = REGISTRY.loads_plan(path.read_text(encoding="utf-8"))
assert restored == WINDOW
```

The dispatcher's `decision.plan` can be saved with the same API. Pass a restored
Plan to `Dispatcher(registry=REGISTRY).run(binding, inputs, plan=restored)`.
Linear can use `LinearBinding(operator, p)` and `LinearInputs(x)`. Compatibility
with the actual device, Tensors and mathematical contract is checked at runtime.

Saved fields are `schema_version`, `algorithm_id`, `algorithm_revision` and
`recipe`. Export/import requires no GPU. Recipes use JSON-native scalars, lists
and string-keyed dictionaries. Reject nonfinite numbers, duplicate JSON keys,
implicit type conversions and unregistered/unvalidated settings. Export returns
a dictionary independent of the original recipe and verifies that the round trip
matches the original Plan. JSON text is deterministic, with sorted keys.
`algorithm_revision` versions the implementation contract. Reproduction also
requires source commits/hashes and execution conditions, as in benchmarks.

## Selectors

See the [selector contract and exact-match artifacts](dispatch/README.md).
Dedicated full/window selection APIs were removed; both are ordinary Algorithm
candidates. Pass a selector to `CSTLinear(selector=...)`, separately from
mathematical declarations and checkpoints.

## Fixed grid and geometry metadata

RegularGrid and Geometry fixed values live in immutable Specs rather than Tensor
buffers. Placement hints before initialization retain no Tensors; after training
starts, atoms/inputs determine execution placement. Dtype conversion does not
quantize metadata values. Checkpoints save and validate all fixed values.

Period Tensors needed for CUDA Polar updates live in a temporary cache in
AlgorithmState.buffers. Rebuild them when the period tuple, Parameter device/dtype
or constant version changes. Compiled launches also check the constant version
and must not continue using changed constants. Fixed-state signatures and native
guards detect changes to Spec values and dictionaries as well.
