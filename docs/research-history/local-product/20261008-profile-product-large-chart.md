# Larger profile-product chart checkpoint

Following PR #67, this branch introduces explicit large Product/input-Strip
matrix and prepared algorithms. Product axes and Strip input accept up to8192
sites, with at most4,194,304 atoms (5% of8192 squared fits this bound). Batch1..64,
FP32 IEEE, regular Euclidean Line axes, Polar[A,4] and two Triweight profiles
remain required. Strip output stays2..128 with existing tile choices. Earlier
algorithms retain their1024-site and Product65536-atom bounds.

The executors and recipe fields are unchanged. Whole-chart normalization and
its single floor, complete support, all canonical gradients, retained-forward
snapshots and live width/pitch updates carry over. This is a scaling experiment;
no GEMM tuning or approximate support is mixed into it. Explicit large IDs use
revision v1 and reuse the strict grouped matrix/preparation recipe types.

CPU1114 passed/1898 skipped, fourteen new CPU declarations/boundaries, eight
case snapshots, prepare/check for both families, Ruff, wheel/sdist build and
isolated installed-wheel metadata imports without Triton/benchmarks passed.
The new48-test suite includes34 actual CUDA cases: full-site FP64 Y/dX/all-atom
gradients at2049x2051 and8192x8192 Product /8191-input partial Strip, full/support
preparation, both matrix engines and the factor control, saved-forward changes,
twenty captured optimizer updates, dynamic pitch, and sort/view/inverse keys
above the int32 limit. Existing regression suites accompany the GPU gate.

GPU correctness and complete-step performance are pending at this source
checkpoint. No large performance or default dispatch adoption is claimed.
Raw CPU/build logs and selected GPU drivers remain in ignored
benchmarks/cuda/linear/evidence/profile-product-large-chart-20261008/.
