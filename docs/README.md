# Documentation

The root [README](../README.md) defines the public API. This directory holds
cross-cutting design notes for charts, coordinates, and optimizers.

- [Chart geometry](chart-geometry.ja.md)
- [Direct amplitude and bandwidth coordinates](direct-amplitude-bandwidth.ja.md)
- [CSTOptimizer](cst-optimizer.ja.md)

Implementation notes live beside the relevant code:

- [Declaration and execution layout](../src/torchcst/_backends/README.md)
- [Kernel and Profile declarations](../src/torchcst/kernels/README.md)
- [Geometry and Chart declarations](../src/torchcst/geometry/README.md)

- [CUDA backend and dispatch notes](../src/torchcst/nn/_backends/notes/README.md)
- [CUDA Linear experiments and kernel decisions](../experiments/cuda/linear/README.md)
- [CUDA Linear benchmarks and measurements](../benchmarks/cuda/linear/README.md)

Historical GPU measurements are linked from the experiment and benchmark
directories. An old result does not by itself change the public backend policy.

- [Operator の宣言と実状態](../src/torchcst/operators/README.md)
