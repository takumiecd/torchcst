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

The GPU gate at source fb9c79dbb9824c37b82259708231815eb48f8efa subsequently
passed861 tests: large48, matrix596, preparation75, grouped56, global30 and
Strip56 (726 actual CUDA,135 CPU/metadata). Job:
l4job-d347d87e317e4f47ac47b3c63cfe34ed. Its driver argument/proof uses the
unambiguous short commit fb9c79db; the companion JSON resolves the full commit
and verifies exact implementation file sets/bytes against the frozen archive.
Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, NVIDIA L4; no tolerance changes.

Complete-step performance is pending. Two2048 primary jobs are selected with
rho3 and rho8, four isolated routes (prepared, nativeG4P8/G8P8, TorchG8P8),
same seed41/5% atoms/B32 and the preceding production optimizer protocol:
l4job-f8f23b04647d44e5aa44aedcf87a30bd (Product) and
l4job-bb56666e248047169c46bbc03923223f (Strip). Each rho has a predeclared600s
driver child timeout; the family job has1300s for both rhos/setup. A separate
job l4job-3e9ac4544164476887606ddc5491360e checks the exact full8192 Product
oracle's cost and native Y/dX/all3,355,443 atom gradients, with default FP64
chunk512 and no sampling. Its first-call cost/validation peaks are diagnostics,
not uninstrumented complete-step time or capture/replay memory. Timeout900s.
The single-worker shared pool serializes this selected batch. Its active VM
will be stopped by the supervisor after the batch; this is not a shutdown claim.

No large performance or default dispatch adoption is claimed yet. Source stays
on kernel/profile-product-large-chart until measured validation passes.
Raw CPU/build logs and selected GPU drivers remain in ignored
benchmarks/cuda/linear/evidence/profile-product-large-chart-20261008/.
