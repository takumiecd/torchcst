# Radius cell-mass Conv2d on RTX 3090

**English** | [日本語](radius-cell-mass-rtx3090-20261124.ja.md)

The RigL comparison will use radius cell mass. This measurement uses
Chebyshev M8 and rational radius coordinates. The Torch reference computes
integrated cell masses over each axis, contracts the factors as
`W = F_out @ F_in.T`, then applies PyTorch Conv2d. Backward differentiates
through the same cell integrals and rational radius law. Initialization is
on CPU, followed by moving the layer to CUDA for eager execution.

The fixture uses an output Chart of 128 sites, with base tile 1, and an input
Chart of 96 sites for 1×1 or 864 sites for 3×3, with base tiles 3 and 27
respectively. Both axes use unit spacing and merge factor 2. Each site count
divided by the base tile is a power of two. The charts are periodic, and the
input Chart flattens channel, kernel height and kernel width in that order.
The rational radius cap is 32, and the initial learnable fraction is 0.15.
The radii, amplitudes and centers take optimizer updates throughout the run.

Two RTX 3090 processes (PIDs 3879 and 3891) used synthetic FP32 channels-last
input `[256, 96, 8, 8]`, stride 2, one group, 1,774 Atoms and fixed dY.
Padding was 0 for 1×1 and 1 for 3×3. Torch 2.6.0+cu126, CUDA 12.6,
cuDNN 9.5.1 and driver 580.95.05 were used with TF32 disabled. Within each seed,
all CST arms shared the initial Atom rows, input and dY. Each arm included
forward, backward with dX, and its own fused CSTAdamW or AdamW with learning
rate 0.001 and zero weight decay. The table shows the range of CUDA-event
medians from the two independent processes with reversed arm order. Each
process used 8 warmups and 24 samples per condition. Allocated and reserved
are total PyTorch allocator peaks, including live allocations.

| Spatial kernel | Backend | Layer step (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | --- | ---: | ---: | ---: |
| 1×1 | Dense | 0.275–0.300 | 40.52 | 108 |
| 1×1 | Materialized | 12.940–14.852 | 222.54–222.59 | 352 |
| 1×1 | Factored | 12.928–12.984 | 134.16–134.18 | 176 |
| 1×1 | Composed | 12.945–15.366 | 61.68–61.73 | 118 |
| 3×3 | Dense | 1.002–1.012 | 42.45 | 90 |
| 3×3 | Materialized | 17.174–17.509 | 2323.93–2323.96 | 2456 |
| 3×3 | Factored | 23.415–23.426 | 2838.18–2838.21 | 2860 |
| 3×3 | Composed | 11.408–12.059 | 107.63–107.66 | 130 |

Independent scalar quadrature and finite differences validated Y, dX and all
four Atom gradient fields in small 1×1 and 3×3 fixtures on CPU and CUDA.
The performance fixture also checked Y, dX and each Atom gradient field against
the CPU FP64 reference before timing.
The composed maximum relative L2 error was 5.09e-7 across all checked tensors
and both seeds. Each arm reached 32 optimizer steps. The performance evidence
covers the isolated FP32 layer. Trained checkpoints, BF16-autocast training,
CUDA Graphs, process-level memory and dispatcher admission remain open.

The 3×3 result establishes composed as the stronger reference for this
fixture. On 1×1 it reduces peak memory and remains close to or slower than
factored on time. For the next CUDA candidate, reuse the Linear formulation
of separate weight assembly and the `dW → dAtoms` pullback, with independent
input/output Fourier routing. The existing square wide recipe has different
axis restrictions. The cell-mass Conv2d route uses the Torch reference;
profile factor assembly and the pullback before tuning them. The
[periodic point-Gaussian result](composed-weight-rtx3090-20261124.md) has a
different profile and separate timing data.

## Reproduction

The measured source is `1f6389c17d55e2f3baeb64ffc3458cc0f2c84155`.
The driver, source archive and SHA-verified results are in the worktree's
ignored `output/conv-radius-20261124/` and the Kyutech home-relative
`cst-conv-radius-20261124-1f6389c1/`. With the source extracted into
`source/`, run these commands from the driver directory.

```sh
PYTHONPATH=source/src python -m pytest -q source/tests/test_cst_conv_radius_cell_mass.py
OMP_NUM_THREADS=2 PYTHONPATH=source/src python measure.py --seed 17 --order dense,materialized,factored,composed --source-commit 1f6389c17d55e2f3baeb64ffc3458cc0f2c84155 --output run-17.json
OMP_NUM_THREADS=2 PYTHONPATH=source/src python measure.py --seed 23 --order composed,factored,materialized,dense --source-commit 1f6389c17d55e2f3baeb64ffc3458cc0f2c84155 --output run-23.json
```

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `bace2d55d9e4a69e15818dc3eeb9be508bf1d5290b970f57c84b62c07fe40652` |
| `measure.py` | `99fd804339e3ef7717beb88fa226c3b533de45cd0bdd4b1a1ca267bb524bcb99` |
| `run-17.json` | `37fc3f2ff818cfcb45b7e7a7cf881e2ef1629dfaabd1ad4c84a8b9ca9ec9a29e` |
| `run-23.json` | `794344df8cabee76ed6bc0141cb1c6bdd649a33975d05923b5fde03e198131b3` |
