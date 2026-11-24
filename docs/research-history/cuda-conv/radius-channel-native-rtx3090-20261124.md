# Native radius cell-mass channel weights on RTX 3090

**English** | [日本語](radius-channel-native-rtx3090-20261124.ja.md)

This experiment implements the [channel-chart design](radius-cell-mass-channel-charts-20261124.md).
Each spatial tap owns its Atom rows. The private `channel_weights` operation
accepts contiguous FP32 physical rows with shape `[tap, atoms, 4]` and
returns `[tap, 128, input_sites]`. Both axes use M8, unit spacing, base tile
1, merge factor 2 and a cap of 32 sites. The existing physical decoder
maps rational amplitude and fraction to radius, and its autograd chain
supplies the rational gradients. Ordinary Conv2d consumes the assembled
filter and supplies the weight cotangent. The native adjoint reads that
cotangent and the saved forward geometry.

The capped parser and weight-only entry are scoped to this research route.
The direct implementation launches across all taps together. The Fourier
implementation reuses the Linear geometry, owner-pair accumulation, fold,
and cotangent contractions and runs the contractions per tap. Levels 3, 4 and
5 start Fourier accumulation above radii 4, 8 and 16, respectively. Level 6
uses direct accumulation at every reachable level.

## Paired training-step measurements

Two fresh RTX 3090 processes (PIDs 4400 and 4415) compared the same filters
and input within each seed, with reversed backend order. Input was synthetic
FP32 channels-last `[256, 96, 8, 8]`, stride 2, one group, output width 128,
and padding 0 or 1 for the 1×1 and 3×3 cases. Each case had 1,773 Atoms,
197 per tap for 3×3. The initial radius-fraction distribution was uniform on
[0, 0.25] or [0.65, 0.98]. These produced approximate physical range
1.44–6.71 and 11.32–30.68. Each step included rational decoding, weight
construction and layout conversion, forward, backward with dX and a fixed
dY, fused AdamW (lr 0.001, decay 0) and amplitude/fraction projection for
the Atom arms. Dense started from the corresponding represented weight and
used ordinary AdamW. The Torch reference used the same capped channel
charts and a batched factor product. Each arm ran 8 warmups and 24 measured
steps. TF32 was disabled with CUDA 12.6, driver 580.95.05, cuDNN 9.5.1 and
PyTorch 2.6.0+cu126.

| Shape and radius range | Dense | Torch | CUDA direct | Fourier 3 | Fourier 4 | Fourier 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1×1 narrow | 0.278–0.289 | 15.405–15.437 | 1.562–1.704 | 1.860–1.891 | 1.793–1.815 | 1.734–1.765 |
| 1×1 wide | 0.285–0.393 | 11.415–11.935 | 2.419–2.561 | 1.886–2.580 | 1.735–1.816 | 1.748–1.767 |
| 3×3 narrow | 1.011–1.012 | 14.785–14.818 | 1.600–1.612 | 3.650–3.665 | 3.527–3.551 | 3.280–3.327 |
| 3×3 wide | 0.998–1.002 | 10.802–10.976 | 2.811–2.841 | 3.634–4.363 | 3.141–3.156 | 5.257–5.273 |

The times are ranges of CUDA-event medians in ms. Synchronized wall time
and all raw samples are in the JSON. For 3×3 the direct path used
42.53–42.57 MiB allocated and 62 MiB reserved. The Torch reference used
63.33–63.35 MiB allocated for narrow and 110.72–110.74 MiB for wide, with
76 and 134 MiB reserved. Dense used 43.01–43.05 MiB allocated and 64 MiB
reserved. These are total allocator peaks with the live fixture included.
Process-level memory was unmeasured.

The trial supports direct for 1×1 narrow and for both 3×3 conditions.
For 1×1 wide, both Fourier levels 4 and 5 beat direct in both processes.
Their relative order varied. The 3×3 Fourier result is for per-tap
launches, so batching the Fourier contractions across taps is a separate
performance question. Public selection needs evidence on trained checkpoints,
model integration and the applicable arithmetic modes.

## Correctness and reproduction

Checked W, Y, dX and all four rational Atom fields against the CPU FP64
reference on the timing fixtures. Maximum relative L2 error across native
paths was 1.84e-6. Targeted tests check scalar Chebyshev quadrature,
physical radius boundaries and periodic owners, and all first-order gradients
at input widths 96 and 128 for both spatial shapes. CUDA Graph replay covers
changed inputs, centers and radius banks with prepared physical rows. The
expanded suite passed 19 tests. The measured rational decoder and optimizer
step ran eagerly. The two existing 2048-wide Linear gradient tests also passed
with the modified extension. The CPU suite passed with 2,219 passed and 2,605
skipped; the wheel and source distribution built.

Measured runtime source `ded7540a086a0500a7288931080e0a6daaf3c539`.
The source, logs, driver and results are in the worktree's ignored
`output/conv-radius-native-20261124/` and Kyutech home-relative
`cst-conv-channel-20261124/ded7540a/`. The isolated build used Ninja 1.13.2,
CUDA architecture 8.6 and the CUDA headers in the installed NVIDIA Python
packages. After extracting the source and driver and configuring these
headers and Ninja, run from the driver directory.

```sh
export PYTHONPATH=source/src:source:source/tests TORCH_CUDA_ARCH_LIST=8.6
python -m pytest -q test_radius_channel_weight_cuda.py
python measure.py --seed 17 --source-commit ded7540a086a0500a7288931080e0a6daaf3c539 --output run-17.json
python measure.py --seed 23 --source-commit ded7540a086a0500a7288931080e0a6daaf3c539 --output run-23.json
```

For the expanded suite, run the same pytest command against
`test_radius_channel_weight_v2.py` in the evidence directory.

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `861c17fabac10a812a637b6898d7f0ad30487259e35d3ca16414525b4d47b1f6` |
| `driver.tar.gz` | `8079e79d12063243e8524fd1738f83529d5a9a137662c46682fe443a040143ba` |
| `measure.py` | `9f60b544e9534e3c897d5d0c687bed4564fd3e175a6939a30fdd3ea11c0f3873` |
| `test_radius_channel_weight_v2.py` | `dcacc91d42b03b8fb64d472557531fb5dcfe88d07d3ede3097178c915ff756db` |
| `run-17.json` | `fb24f893d000a00d84bf791de95eb9dbc7a7d34c0dcc111c7cbf330a32b3263e` |
| `run-23.json` | `b9a9c2e73a5541940aed53767dc0eef0aa5cf3785c45283e92d0ca5e17a693f0` |
