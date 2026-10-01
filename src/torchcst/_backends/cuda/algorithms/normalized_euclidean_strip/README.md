# Globally normalized Strip execution

`torchcst.nn.NormalizedStripLinear` owns signed amplitude, log bandwidth and
three continuous center coordinates. Both implementations evaluate joint 3D
Triweight support and its discrete L2 norm over the whole finite operator,
with norm floor `1e-6` and bandwidth clamped to `.03–3.25`.

- `full/executor.py` and `full/kernels.py`: default AtlasStream implementation. Retains the whole FP32 weight
  and forms its adjoint for the complete all-parameter VJP.
- `window/executor.py`, `window/provider.py` and `window/kernels.py`: opt-in rolling implementation. Reuses a
  `min(512, N) × K` FP32 weight/adjoint buffer and an `8 × K` halo. The global
  norm, current support and row buckets are rebuilt each forward. Backward
  rebuilds weights for dX and processes each atom's complete coupled VJP once.
- `_shared/atlas_table.py` and `window/narrow_table.py`: conservative candidate tables, not
  fixed-center approximations. Actual site values use current parameters.
- `_shared/common.py`: support offsets, guards and GPU bucket permutation.

CUDA uses FP32, spacing `(1, .5, .5)`, one atom-program warp and FP fusion.
Window routing also requires quarter-grid origins and rows divisible by 32;
otherwise the validated full route is used. Coordinate guards reject ranges
outside the demonstrated FP32 routing regime. Atomic accumulation is not
bitwise deterministic. Autocast and TF32 are rejected. CUDA supports first
derivatives; CPU uses `_backends/torch/operators/normalized_radial.py`.

The independent oracle and full/window Graph/optimizer integration check are
in `tests/test_normalized_strip_public.py` and
`benchmarks/cuda/linear/check_normalized_strip_public.py`. Historical
measurements and their limits are recorded in
`benchmarks/cuda/linear/results/normalized-strip-history-20261001.json`.
See `docs/normalized-strip.ja.md` for usage and measured tradeoffs. CPU imports
need no Triton; all CUDA modules are loaded lazily and included in the wheel.

`full/algorithm.py` and `window/algorithm.py` contain metadata-only Algorithm
registrations, with launch settings in each `recipe.py`. The registry consumes
the common `operators.OperatorSpec`; `contract.py` checks the fixed kernel and
Euclidean site contract. Importing registration metadata never imports Triton.
