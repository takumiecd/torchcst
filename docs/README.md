# Documentation

The root [README](../README.md) defines the public API. This directory holds
cross-cutting design notes for charts, coordinates, and optimizers.

- [Chart geometry](chart-geometry.ja.md)
- [Direct amplitude and bandwidth coordinates](direct-amplitude-bandwidth.ja.md)
- [Normalized optimizer](normalized-optimizer.ja.md)
- [Optimizer selection](optimizer-selection.ja.md)
- [Parameter Adam](parameter-adam.ja.md)
- [Implicit projected moments](implicit-projected-moment-transport.ja.md) and [decisions](implicit-projected-adam-decisions.ja.md)

Implementation notes live beside the relevant code:

- [CUDA backend and dispatch notes](../src/torchcst/nn/_backends/notes/README.md)
- [CUDA Linear experiments and kernel decisions](../experiments/cuda/linear/README.md)
- [CUDA Linear benchmarks and measurements](../benchmarks/cuda/linear/README.md)

Historical GPU measurements are linked from the experiment and benchmark
directories. An old result does not by itself change the public backend policy.
