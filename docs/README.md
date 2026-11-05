# Documentation

The root [README](../README.md) defines the current public API. Code-adjacent
READMEs describe the current declaration, state and execution boundaries.
Current guides use English as the canonical language, with corresponding Japanese
versions linked below. Keep both versions equivalent when updating a guide.

## Current contracts

- [Declaration and execution layout](../src/torchcst/_backends/README.md)
- [Kernel: profile, factor and coordinate composition](../src/torchcst/kernels/README.md)
- [Geometry metadata](../src/torchcst/geometry/README.md)
- [Chart declarations and owners](../src/torchcst/charts/README.md)
- [Regular grids, weight axes and fixed metadata](regular-grid.md) ([日本語](regular-grid.ja.md))
- [Single-Chart Polar profile products](profile-product.md) ([日本語](profile-product.ja.md))
- [Pattern declarations](../src/torchcst/patterns/README.md)
- [Operator declarations and live state](../src/torchcst/operators/README.md)
- [CSTAdamW and CSTOptimizer: training, decay and checkpoints](cst-optimizer.md) ([日本語](cst-optimizer.ja.md))
- [Radius cell masses: CPU example, hierarchy and CUDA Plans](radius-cell-mass.md) ([日本語](radius-cell-mass.ja.md))
- [Execution, Plan selection and dispatch boundaries](algorithm-dispatch-boundaries.md) ([日本語](algorithm-dispatch-boundaries.ja.md))
- [Atom and Algorithm state ownership](atom-state.md) ([日本語](atom-state.ja.md))

## Development and measurements

Development starts at [CONTRIBUTING](../CONTRIBUTING.md), the canonical English
guide ([日本語](../CONTRIBUTING.ja.md)).

- [CUDA kernel development workflow and examples](kernel-development.md) ([日本語](kernel-development.ja.md))
- [Owner research resources and operating rules](research-operations.md) ([日本語](research-operations.ja.md))
- [CUDA Linear benchmarks and measurements](../benchmarks/cuda/linear/README.md)

## Model-level CST research

[cst-research](https://github.com/takumiecd/cst-research) organizes research
questions, study protocols, dense/DST comparisons, learning curves and result
indexes. Studies pin the torchcst revision they use and distinguish exploratory
results from confirmation. Current experiment sources and older evidence remain
at their recorded locations while the research tooling is migrated.

Operator/kernel development and correctness validation stay in torchcst.
Research-specific private execution paths are not public API guarantees.

## Validation and revision-specific design notes

These notes retain their recorded design, API and measurement scope. Older API
examples are historical; use the root README and current contracts above for new code.
An old GPU result does not establish performance or dispatch policy for a new source.

- [Kernel API unification and Step 3 CPU/L4 validation](research-history/api-unification-step3-20261102.ja.md)
- [Tensor-free coordinate metadata and Step 4 CPU/L4 validation](research-history/static-coordinate-metadata-20261102.ja.md)
- [Chart geometry](chart-geometry.ja.md)
- [Direct amplitude and bandwidth coordinates](direct-amplitude-bandwidth.ja.md)
- [Kernel cleanup and validation](kernel-cleanup.ja.md)
- [Chart and Geometry cleanup and validation](chart-geometry-cleanup.ja.md)
- [Optimizer replacement and recorded validation](cst-optimizer-migration.ja.md)
- [Radius cell-mass design and measured CUDA scopes](cell-mass-kernel-flow.ja.md)
- [Normalized Strip design and recorded CUDA scopes](normalized-strip.md) ([日本語](normalized-strip.ja.md))
- [H generation, reuse and lifetime: kernel design policy](h-lifecycle-kernel-design.md) ([日本語](h-lifecycle-kernel-design.ja.md))
- [CUDA backend migration](cuda-backend-migration.ja.md)
- [CUDA backend and dispatch history](backend-history/README.md)
- [Historical CUDA Linear research and decisions](research-history/cuda-linear/README.md)
- [Benchmark cleanup and validation](benchmark-cleanup.ja.md)
- [CSTLinear entry points and reuse of existing atoms](linear-unification.ja.md)
- [Concrete Chart types and Geometry / Pattern separation](chart-types.ja.md)
