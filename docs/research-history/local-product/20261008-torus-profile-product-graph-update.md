# Intrinsic Torus profile-product Graph update

Follow the owner-selected centre-fibre S1xS2 profile-product revision2.
This branch builds on reference PR #71 and adds an explicit research-only
Torch atom_update Algorithm, not a new public optimizer default.
Intrinsic [A,5] FP32/FP64 only; ambient/older metrics/low precision are rejected.
The common Dispatcher/AtomUpdateBinding owns no optimizer state. AdamW retains
its proposal, moments and clocks. No linear-specific updater in the optimizer.

Reuse the exact Polar activity law; express the public intrinsic retraction
with device operations: limit arc displacement, periodic wrap, and cap section
normal coordinates below the antipodal margin. Keep R/r live. The eager law's
Python-double multiplication pi*R is reproduced in FP64 before casting each
operand to the parameter dtype. No Tensor-to-host reads during capture.

CPU gate:13 passed/20 CUDA cases skipped. Covers both activity modes,
FP32/FP64, three live major radii, minor-radius changes, zero/under/over-radius
Polar points, arc crossings/cap and section cap; update matches the public law
bit-for-bit. CUDA gates additionally compare20 captured AdamW updates/moments
and complete Y/dX/dP/parameter steps with evolving widths, Product/Strip and
reversed logical axes. Strict optimizer and numerical checks remain unchanged.

The ordinary CSTOptimizer wrapper still requires eager execution; the research
benchmark uses an explicit base-optimizer proposal plus declared update Plan,
as existing Euclidean complete-step measurements do. Intrinsic tangent/state
transport is identity. Ambient GPU/Graph updates are outside this candidate.

L4 GPU gate:33 passed, zero skipped,6.06s. This includes both activity modes,
strict FP32/FP64 optimizer state equivalence and20 captured full steps with
evolving widths in Product/Strip/reversed axes. No tolerances were relaxed.
Complete-step time/peaks remain unmeasured. This provides a verified research
Graph comparison baseline before fused CUDA work; it establishes correctness.

- Job:l4job-d1b6c6d403034932b19bf5676b256796
- Measured commit:d1dd00edff1b99153b4c7bd3d16e83b8305b7709
- Frozen source SHA256:1efbcd997bf27bdc3c2afe816758512a4c419a9a89dddfe6d13bd80ce2631985
- Result archive SHA256:fc230dfa062c91586ff595e2d9f0a1ed8ca2051aa5dfcf5c30b65dfdb763a335
- Runtime:NVIDIA L4,Torch2.11.0+cu130,CUDA13.0,Triton3.6.0.
- Driver:benchmarks/cuda/linear/evidence/torus-profile-product-graph-update-20261008/driver.py
- Reproduction:pytest -q tests/test_torus_profile_product_graph_update.py

The branch was subsequently rebased on the reference validation note9e369267;
the updater and test contents are identical to the measured source checkpoint,
which is also preserved in kernel/torus-profile-product-graph-update-first-gate.
The shared pool keeps source.tar.gz,spec.json,receipt.json,results.tar.gz and
results/artifacts/{graph-update.log,gate-proof.json} for this job under
~/.local/state/colab-l4-pool/jobs/. The local archive hash matches the receipt.
Raw drivers/logs/results remain in ignored evidence/ and the shared pool.
