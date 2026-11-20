# Streamed 1×1 periodic Gaussian shortcut on RTX 4090

**English** | [日本語](pointwise-gaussian-20261118.ja.md)

This is the CIFAR-10 two-CST-head model's last shortcut. The fixture restores the
seed-17 checkpoint and regenerates the earlier qualification batch and
cotangent. The channels-last input is `[256, 96, 8, 8]`, with stride 2, zero
padding and one group. The output is `[256, 128, 4, 4]`, or 4,096 rows. All CST
arms use the same 1,774 atoms, gain and L2-normalized periodic point Gaussian
with fixed σ=0.75 on both axes. The independent correctness check also uses the
saved seed-17 trained shortcut atoms and their gain, with the same input.

## Dataflow

The Torch factor preparation used by CSTLinear and CSTConv2d supplies
`F_in[C_in,A]` and `F_out[C_out,A]`, with amplitude in the output factor.
The CUDA path packs the selected pixels as rows. Each Triton block applies
both factors for up to 32 rows and 32 atoms at a time. Its atom response stays
inside the block. Each atom split produces a partial output, which is then
summed. The selected setting uses four splits and 8 MiB of transient forward
partial outputs for this fixture. The factored path saved a 27.72 MiB
`[256, 1774, 4, 4]` activation for backward. Autograd hooks confirm that the
streamed path saves the sampled input rows and factors in place of that
activation. The bound of four partial outputs is independent of the atom count.

Backward reuses the forward factors, interchanges them for dX, and scatters
dX into the input grid with zeros at skipped sites. Each output stripe forms
`dY_stripe.T @ X` and immediately contracts it with both factors for the
factor VJPs. The largest cotangent stripe in this fixture is 32×96 scalars.
The existing factor autograd supplies the amplitude and center gradients.
The input packing, output layout conversion, factor preparation and stride
scatter are included in timing.

The earlier Linear packed-levels comparison uses a different M8 cell-mass
profile and admitted dimensions. Its CIFAR-10 middle-layer case was
2,048×2,048 with 256 rows. Its rectangular recipe is limited to specific widths
and at most 256 rows, and the owner-window fold was qualified on N1024 with
at most four rows. The fold identifies equivalent frequencies of the existing
finite, cell-integrated Fourier basis. This point-sampled Gaussian uses its
original periodic distances and normalization.

## Measurement

Two fresh processes with different arm orders ran on the same idle RTX 4090,
CC 8.9, driver 570.86.15, CUDA 12.8, cuDNN 9.10.2, Torch 2.9.1+cu128 and Triton
3.5.1. Each process used ten warmups and 60 synchronized GPU-event samples
per arm. The table shows ranges across the two run medians. All tensors and
accumulators use FP32. PyTorch matmul and cuDNN ran with TF32 disabled. The
streamed dots use the sum of three TF32 products for FP32-like accuracy.
The complete layer step includes dX and CSTAdamW at lr=0.001, or ordinary
AdamW at lr=0.001 for Dense. The externally supplied dY is fixed.

| Path | Forward ms | Backward ms | F+B ms | Layer step ms | Extra forward allocated MiB | Extra F+B peak allocated MiB | Total peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CST two convolutions | 0.994–1.005 | 1.125–1.130 | 2.120–2.136 | 2.238–2.245 | 37.42 | 97.42 | 182 |
| CST composed weight + one convolution | 0.575–0.582 | 0.299–0.305 | 0.874–0.888 | 1.607–1.628 | 7.46 | 23.46 | 62 |
| CST streamed | 0.833–0.839 | 0.404–0.460 | 1.232–1.304 | 1.868–1.896 | 17.29 | 22.71 | 62 |
| Dense FP32 | 0.020 | 0.084–0.089 | 0.103–0.109 | 0.147–0.207 | 2.00 | 18.00 | 54 |

Extra allocated memory subtracts the live allocation immediately before
measurement, 24.35 MiB in both runs. Complete-step extra allocated peaks
match the F+B values at this precision. Reserved memory is the total
PyTorch allocator peak. The comparison measures a layer and uses PyTorch
allocator statistics. The full CST layer runs eagerly because existing
RegularGrid factor preparation checks bounds on the host during capture.
The prepared-factor CUDA path passed Graph replay with changing inputs and
changing factors. Dense F+B Graph capture/replay receipts are in the JSON.

The independent FP64 oracle evaluates the full periodic Gaussian and runs a
dense convolution with autograd. It validates all 4,096 rows and every atom.
The streamed maximum relative L2 errors across both runs and both atom states
were 2.50e-7 for Y, 2.22e-7 for dX, 2.23e-7 for amplitude, 2.40e-7 for the
output center, and 3.28e-7 for the input center. The largest absolute error
was 3.36e-6. CUDA regression tests passed for odd spatial extents, asymmetric
stride, layouts, frozen inputs or atoms, the empty batch and prepared Graph
replay. CUDA tests passed 22/22.

A 12-setting tile screen selected four splits. Seven splits produced 14 MiB
of forward partials. Four reduced that to 8 MiB and had the shortest median
F+B in the screen. The original pipelined launch exceeded Ada's shared
memory limit. The measured launches use one pipeline stage.

## Decision for 3×3

Streamed reduced F+B time by 39–42% and extra peak allocation by 76.7% versus
two convolutions. Composed weight was faster in both runs and used 0.76 MiB
more extra peak allocation over F+B. The streamed path remains an opt-in
research backend. At 96 input channels a 3×3 patch has 864 features. A
follow-up can test implicit patch gathering into channel/offset tiles and
overlap accumulation for dX, with a gate against composed Conv2d on time,
peak and the independent oracle. A flattened 3×3 patch can have a different
input chart, so its factors must follow that chart's declared ordering.

## Reproduction and evidence

The measured source commit is `14b9a7792cd9cffcb5bf1ab2b7b4d7ddee8f24e8`.
The isolated Kyutech job PIDs were 63739 and 63780. The frozen fixture
and original checkpoints remain on the experiment host. The driver, logs,
source archive and result receipts are preserved in ignored
`output/cifar10-cnn-cst-conv-streamed-20261118-01/`.
The driver uses the remote directory `cifar10-cnn-cst-conv-streamed-20261118-01`
under the preceding study's lab root. Put `reproduce.sh`, `freeze_fixture.py`,
`measure.py` and the source archive there. Extract the archive into
`source-v5/`, then run `bash reproduce.sh fixture2` if `frozen.pt` is absent.
The launcher prints the PID and returns. Wait for `frozen.pt` before the first
measurement and for `final_1.json` before starting the second process. Run
`verify.py` locally on the downloaded JSON receipts.

```sh
bash reproduce.sh final_1 factored,composed,streamed,dense_fp32
bash reproduce.sh final_2 dense_fp32,streamed,composed,factored
python3 verify.py final_1.json final_2.json
```

| File | SHA-256 |
| --- | --- |
| `frozen.pt` | `d725da274f105c22aece28bfa89856953a41922d9bce93093f2f6ea342f7dd85` |
| `source-v5.tar.gz` | `1e094d11a7803a7650f2610528968381bda9abdc71bfe77f7435d1f98b84c595` |
| `measure.py` | `ae81dcf760fdf87f53df88e6a64c857fa00921bd7655f6b27ce63b87c5a63e46` |
| `final_1.json` | `57e8ee812b6347fb258206a857489d9c002bfe6964e7e1ebda73003d14d0d69c` |
| `final_2.json` | `bb4c1ef8d7f8113d5e95bc9aa7e4ac787e149a6f791c4f06e7b74a853831d930` |
