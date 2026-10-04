# Documentation

The root [README](../README.md) defines the public API. This directory holds
cross-cutting design notes for charts, coordinates, and optimizers.

- [Chart geometry](chart-geometry.ja.md)
- [Direct amplitude and bandwidth coordinates](direct-amplitude-bandwidth.ja.md)
- [CSTOptimizer](cst-optimizer.ja.md)

Implementation notes live beside the relevant code:

- [Declaration and execution layout](../src/torchcst/_backends/README.md)
- [Kernel and Profile declarations](../src/torchcst/kernels/README.md)
- [Kernel cleanup and validation](kernel-cleanup.ja.md)
- [Geometry declarations](../src/torchcst/geometry/README.md)
- [Chart declarations and owners](../src/torchcst/charts/README.md)
- [Pattern declarations](../src/torchcst/patterns/README.md)
- [Chart and Geometry cleanup and validation](chart-geometry-cleanup.ja.md)

- [CUDA backend migration](cuda-backend-migration.ja.md)
- [CUDA backend and dispatch history](backend-history/README.md)
- [Historical CUDA Linear research and decisions](research-history/cuda-linear/README.md)
- [CUDA Linear benchmarks and measurements](../benchmarks/cuda/linear/README.md)
- [CUDA kernel development workflow and tools](kernel-development.ja.md)
- [Benchmark cleanup and validation](benchmark-cleanup.ja.md)

Historical GPU measurements are retained under research-history and benchmark
results. An old result does not by itself change the public backend policy.

- [Operator の宣言と実状態](../src/torchcst/operators/README.md)

- [CSTLinear の入口・既存 atom 再利用の統一](linear-unification.ja.md)

- [Chart の具体型と Geometry / Pattern のパッケージ分離](chart-types.ja.md)
