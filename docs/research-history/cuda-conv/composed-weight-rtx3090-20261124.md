# Composed Conv2d weights on RTX 3090

**English** | [日本語](composed-weight-rtx3090-20261124.ja.md)

`CSTConv2d(backend="composed")` computes `W = F_out @ F_in.T` and applies
PyTorch Conv2d. PyTorch differentiates through both operations. The final
weight is reused for all spatial positions in the batch. The existing
two-convolution path applies an intermediate convolution with one output
channel per Atom. The existing materialized path builds one complete
weight tensor per Atom before reduction.

Two RTX 3090 processes (PIDs 3763 and 3787) used synthetic channels-last
FP32 input `[256, 96, 8, 8]`, output `[256, 128, 4, 4]`, stride 2, one
group, and 1,774 Atoms. The 1×1 condition used zero padding. The 3×3
condition used padding 1 and an 864-site flattened input Chart. Both used
L2-normalized periodic Gaussian factors with fixed σ=0.75 and trained
amplitude and centers. Each process shared input, cotangent and initial
Atom parameters across CST arms. Each arm used a separate optimizer and
started from the same initial effective weight as the others within the
numerical tolerance. The seeds were 17 and 23, and the arm orders were
reversed. The GPU had compute capability 8.6, driver 580.95.05, CUDA 12.6,
cuDNN 9.5.1 and PyTorch 2.6.0+cu126, with TF32 disabled.

The eager layer step included forward, backward with fixed dY, dX and
fused CSTAdamW or AdamW at learning rate 0.001 and zero weight decay. Each
arm used 8 warmups and 24 synchronized samples. The table shows the two
CUDA-event medians and total allocator peaks, including live allocations.
Synchronized wall-clock medians are in the receipts. GPU process memory
was unmeasured.

| Spatial kernel | Backend | Layer step (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | --- | ---: | ---: | ---: |
| 1×1 | Dense | 0.264–0.310 | 40.51 | 62 |
| 1×1 | Materialized | 3.269–3.290 | 296.86 | 320 |
| 1×1 | Factored | 3.539–3.780 | 117.87 | 190 |
| 1×1 | Composed | 2.829–2.830 | 45.92 | 70 |
| 3×3 | Dense | 0.967–1.000 | 42.43 | 64 |
| 3×3 | Materialized | 44.347–44.361 | 3046.63 | 3088 |
| 3×3 | Factored | 18.339–18.357 | 2813.72 | 2850 |
| 3×3 | Composed | 3.536–3.539 | 65.57 | 88 |

An independent FP64 periodic Gaussian calculation checked Y, dX,
amplitude and both center gradients at the initial Atom states before
measurement. The largest relative L2 error for composed was 4.57e-7
across both shapes and seeds. All timed arms reached 32 optimizer steps.
The measurements cover an isolated FP32 layer with synthetic inputs.
Trained checkpoints, BF16-autocast training, CUDA Graph execution, and
dispatcher adoption are outside this measurement.

## Reproduction

The measured source is `d8aab933d315d2707b06e7a250756a88049c90bc`. The
driver, source archive and verified receipts are in the worktree's ignored
`output/conv-composed-20261124/` and the Kyutech home-relative directory
`cst-conv-composed-20261124-d8aab933/`. With the extracted source in
`source/`, run the following from the directory containing the driver.

```sh
PYTHONPATH=source/src python measure-v3.py --seed 17 --order dense,materialized,factored,composed --source-commit d8aab933d315d2707b06e7a250756a88049c90bc --output run-17.json
PYTHONPATH=source/src python measure-v3.py --seed 23 --order composed,factored,materialized,dense --source-commit d8aab933d315d2707b06e7a250756a88049c90bc --output run-23.json
```

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `d1f76f5f68d33287c4b582b108755c933a4ce2ee2e773fb3e6907f464f7e64b8` |
| `measure-v3.py` | `f3169d7884149fc161ed2c633e6301a3feb03ac49491ab89a9cde7be69427947` |
| `run-17.json` | `6576d2ae8f5816731ddde1a3249d095057fe1e4a5c69c34633247402b366746e` |
| `run-23.json` | `4032790de620e92d155e75c601d0cb9913a09d9eceec01e9a02ff6fb6251e5ea` |
