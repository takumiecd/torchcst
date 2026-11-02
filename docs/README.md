# Documentation

The root [README](../README.md) defines the current public API. Code-adjacent
READMEs describe the current declaration, state and execution boundaries.

## Current contracts

- [Declaration and execution layout](../src/torchcst/_backends/README.md)
- [Kernel: profile, factor and coordinate composition](../src/torchcst/kernels/README.md)
- [Geometry metadata](../src/torchcst/geometry/README.md)
- [Chart declarations and owners](../src/torchcst/charts/README.md)
- [Regular grids, weight axes and fixed metadata](regular-grid.ja.md)
- [Pattern declarations](../src/torchcst/patterns/README.md)
- [Operator declarations and live state](../src/torchcst/operators/README.md)
- [CSTAdamW and CSTOptimizer: training, decay and checkpoints](cst-optimizer.ja.md)
- [Radius cell masses: CPU example, hierarchy and CUDA Plans](radius-cell-mass.ja.md)
- [Execution, Plan selection and dispatch boundaries](algorithm-dispatch-boundaries.ja.md)
- [Atom and Algorithm state ownership](atom-state.ja.md)

## Development and measurements

Development starts at [CONTRIBUTING](../CONTRIBUTING.md).

- [CUDA kernel development workflow and examples](kernel-development.ja.md)
- [Owner research resources and operating rules](research-operations.ja.md)
- [CUDA Linear benchmarks and measurements](../benchmarks/cuda/linear/README.md)

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
- [Normalized Strip design and recorded CUDA scopes](normalized-strip.ja.md)
- [CUDA backend migration](cuda-backend-migration.ja.md)
- [CUDA backend and dispatch history](backend-history/README.md)
- [Historical CUDA Linear research and decisions](research-history/cuda-linear/README.md)
- [Benchmark cleanup and validation](benchmark-cleanup.ja.md)
- [CSTLinear entry points and reuse of existing atoms](linear-unification.ja.md)
- [Concrete Chart types and Geometry / Pattern separation](chart-types.ja.md)
