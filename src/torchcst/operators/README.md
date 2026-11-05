# Operator declarations and live state

English (canonical) | [日本語](README.ja.md)

A single Chart is the basic layout. The pair of input/output Charts required by
current kernels is also supported. These distinguish observation layouts without
creating two large execution systems.

| Type / field | Meaning |
| --- | --- |
| `SingleChartSpec.chart` | One Chart with logical shape `[out, in]` |
| `ChartPairSpec.input_chart / output_chart` | Independent input/output observations; matrix shape is `[output.features, input.features]` |
| `OperatorSpec.layout` | Exactly one of these layouts; input/output Tensors and the number of Charts are separate concepts |
| `OperatorSpec.kernel` | Atom Profile, amplitude, bandwidth, normalization and update declarations |
| `OperatorSpec.operation_id / revision` | Currently `linear` / `1` |
| `Operator.charts / kernel / atoms` | References to live state owned by existing Modules |
| `Operator.p` | The current `atoms.p` Tensor; no copy or independent Parameter |

A single-Chart kernel is radial or profile_product; a Chart-pair kernel is
separable. An amplitude wrapper applies the same classification to its inner
kernel. Do not reinterpret one layout as the other or duplicate Charts to make
their counts match.

profile_product multiplies coordinate profiles on one Euclidean Product/Strip
grid. Revision 2 `polar_torus_profile_product` multiplies chord-distance profiles
for fibres fixed at the atom center on a single S¹×S² Torus Chart. The backend
separates input/output factors and connects them to single-Chart
`operator.factors()` and Torch factored execution. See the
[mathematical contract](../../../docs/profile-product.md).

```python
from torchcst import CSTLinear

# chart.shape == (out_features, in_features)
single = CSTLinear(chart=chart, atoms=8, kernel=direct_kernel)
single_spec = single.declaration()  # Immutable configuration-time snapshot.
operator = single.operator  # Live view for execution.
y = operator.apply(x)  # Torch materialized reference.

pair = CSTLinear(input_chart, output_chart, atoms=8, kernel=separable_kernel)
y = pair.operator.apply(x, algorithm="factored")

# Check and bind the Spec to existing state at configuration time.
bound = single_spec.bind(
    charts=single.cst_charts(), kernel=single.kernel, atoms=single.atoms
)
```

CSTLinear's Torch materialized/factored implementations, dense weights and atom
evaluation use this entrypoint. Computation lives in
`_backends/torch/operators/linear.py`. Torch autograd tracks inputs, every atom and
trainable Chart Tensors. Passing `p` explicitly also allows derivative-engine
candidate points or atom subsets to be evaluated. `Operator.apply()` uses ordinary
autograd. CSTOptimizer consumes the resulting Parameter gradients without
installing an optimizer-specific backward route.

`Operator` is not an nn.Module. Ownership, checkpoint keys and optimizer Parameter
references remain with the existing Modules. The live view reads current values
after `atoms.p` updates/replacement and dtype/device conversion. CSTLinear
reconstructs the view when a Module itself is replaced. Charts and kernels refer
to concrete ChartState and common KernelState objects. Users provide ChartSpec /
KernelSpec to a layer; new mathematical meanings need corresponding backend
implementations.

`declaration()` and `spec.bind()` operate at configuration boundaries. Do not read
back scalars or snapshot explicit points during forward, backward or CUDA Graph
capture. A trainable-coordinate snapshot fixes the current values; it does not
replace execution Tensors or their gradients. Views use live configuration, so
retrieve a declaration again when using it after a configuration change.

CSTLinear / CSTConv2d construct State from declarations. CSTLinear selects Torch
and CUDA Algorithms through the common Registry / Selector. Register only
Algorithms whose mathematical and numerical contracts match a general
declaration. If Chart pairs are removed in the future, validate normalization
domains and gradients during the migration.

## Common normalized-radial dispatch

`CSTLinear` with `chart` and `kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT` declares a
radial operator on one regular 3D Euclidean Chart. Triweight is L2-normalized over
all operator sites. Its coordinates are signed amplitude, clamped log width and
center, distinct from DirectAmpWidth activity coordinates or per-Chart
normalization.

`atoms=` accepts an integer, Tensor, nn.Parameter or Atoms. A Tensor is detached
and copied; a Parameter registers the same object; Atoms reuses its owner.
Normalization uses the same live Operator / KernelState / ChartState / Atoms.
The public Linear is unified as CSTLinear; old dedicated classes and checkpoint
adapters have been removed.

The common Registry registers Algorithms; a Selector chooses candidate Plans.
The Dispatcher connects Inputs validation, Context construction, support checks,
State retrieval and Algorithm execution. Operator remains the entrypoint to
common Chart/Geometry/Kernel objects; no normalized-specific Operator type or
adapter is added. Normalized Algorithms derive regular-site metadata from
existing declarations. A regular Product and continuous Strip can execute on
the same sites. If the fixed kernel, normalization, bandwidth or Euclidean-site
semantics do not match, use the general Torch reference rather than this
specialized Algorithm.

`CSTLinear(selector=...)` is an execution policy, excluded from OperatorSpec and
checkpoints. Full/window are ordinary registered Algorithms described by Plans.
Launch settings belong in each Algorithm's Recipe. Snapshot the corresponding
fixed metadata at configuration boundaries and reconstruct it after buffer
replacement, version or dtype/device changes. Atom values always use the current
Tensor; do not snapshot them in ordinary forward, backward or Graph capture.
After changing metadata, run one forward outside capture.

ChartSpec / ChartState are ABCs. Their concrete types live in charts/, and layers
construct State with compile_chart. Geometry and Pattern live in geometry/ and
patterns/ respectively.
