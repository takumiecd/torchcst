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
