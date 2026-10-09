# Torus centre-fibre sparse W candidate

The mathematical operator remains the intrinsic S¹×S² centre-fibre profile
product on one circle-output Strip, Triweight/Triweight, live Polar width,
combined complete-chart L2 norm with one floor. This research candidate packs
complete raw supports and norm derivatives once, assembles W through sparse
atom patches, computes Y/dX/shared dW with IEEE FP32 cuBLAS, and contracts every
atom VJP against dW. Coupled section q0 / circle-radius differentiation remains.

For regular circle sites a conservative angular chord bound defines a compact
index window. Positive spacing, full tiles, nonoverlapping pitch, sufficient
capacity, and a period/pitch discrepancy below one spacing are required. The
bound includes the entire pitch gap and two rounding indices; duplicate windows
are disabled when N is below capacity. Unsafe geometry uses complete-axis
packing. Packing overflow uses complete-axis contraction, never truncation.
The S² axis is always fully enumerated during packing. Forward snapshots own
p/scalars/queries/support and W; no forward state lives on Algorithm instances.

Predeclared stage: one 900-driver-second correctness job, 40 GPU tests plus
full-shape B32/N1024/2048/sigma3/8 all-atom physical FP64 oracle and public
optimizer update checks. max absolute, relative L2 and existing elementwise
4e-4 gates; public parameter update 2e-6; twenty captured evolving steps with
moments 2e-5. Gate failure prevents performance measurement. New corrective
source, if needed, is a separate job; failures and scope remain recorded.

Only after correctness, one primary performance job up to 1500 driver seconds
for all four cases, old W+GEMM baseline, sparse W, dense. Existing runner retains
21 full evolving-step samples, eager and Graph, capture/replay peak allocated
and reserved bytes. Process GPU memory is unmeasured. A separate inverse job
(up to 1500 seconds) is eligible only for >3% time improvement or >=5% allocated
memory reduction with <=3% time regression. CPU declarations are not GPU PASS.
Public default selector remains unchanged. Raw drivers/results belong in
ignored `output/torus-sparse-weight/` and the shared pool job directories.

Initial source f6a140b7 gate job `l4job-9e2177a6458a458995b2d795db1e81b7`
failed in driver setup before importing Torch or running any GPU test. The
Colab Python3.13 environment cannot bootstrap venv ensurepip; driver time
0.164 seconds. No numerical or performance result exists. Source archive
`fa62f19dbf1bed3715462043581fc842614bef1e97cce0894787361a5cbc497c`,
result archive `d691537135094321aacf98a8b948203c1a4ab386a5880453953d1628f62a25cb`.
Complete job copied to ignored `output/torus-sparse-weight/failed-setup-v1/`,
all files and archives hash-verified. Corrected driver uses installed pytest
first; only if absent, creates a no-ensurepip job-local environment and directs
pip explicitly to that environment. Corrected gate has 899 seconds, retaining
all tests, four cases, oracle and tolerance gates. This is a setup repair,
not a retry justified by numerical or performance outcomes.

Metadata checkpoint e1f6506 additionally rejects TF32 for cuBLAS contractions;
cohort already sets TF32 false and numerical execution files are unchanged.
Local CPU full suite: 1350 passed, 2418 skipped, 18 warnings,26.44 seconds;
wheel/sdist offline build and Ruff passed. CUDA skips are unvalidated.

Setup-repaired source3992aac job `l4job-474d449075174d26abcc402c59115c9d`
reached GPU tests:37failed/4passed/0skipped in11.50 seconds,16.046 driver
seconds. Nonempty cases failed Triton compilation because the circle pack's
conditional branches reused vector names at different CAP/full-axis shapes;
no numerical comparison or full-atom case completed. Entire job is retained in
`output/torus-sparse-weight/failed-compile-v1/` with every file hash verified.
Correction assigns branch-exclusive names to full-axis vectors, preserving
operations, norm, support, derivatives and all gates. Corrected gate has882
seconds, within the original900 after both failures; no performance runs have
started. Failed source/result hashes remain in the copiedspec/receipt.
