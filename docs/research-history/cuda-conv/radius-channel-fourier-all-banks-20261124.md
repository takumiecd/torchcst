# All-bank Fourier channel weights for Conv2d

**English** | [日本語](radius-channel-fourier-all-banks-20261124.ja.md)

The [previous implementation](radius-channel-native-rtx3090-20261124.md)
uses Fourier coefficients to integrate the cells in every mode. Its direct
mode evaluates the two local cell-mass vectors and adds each Atom's outer
product into the weight. Its hybrid mode aggregates the wider windows in
Fourier coefficients and runs that aggregation per spatial tap.

The new `channel_fourier_weights` entry aggregates the Fourier coefficients
for every reachable bank, including the smallest two-cell period. The
coordinate contract remains periodic per channel axis, Chebyshev M8, unit
spacing, base tile 1, merge factor 2 and radius at most 32. The spatial
taps have separate Atom rows. Ordinary Conv2d applies the final weight.

1. Integrate the original harmonics over whole cells. Shift the integrated
   Fourier coefficients to the owner tile. At integer sites, frequencies
   separated by the local period in sites give the same samples. Combine
   those coefficients, including the signed sine aliases and the even-period
   Nyquist term, giving at most `min(17, period_sites)` real coefficients.
2. Accumulate Atom coefficient outer products by tap, radius bank, and
   periodic input/output owner. The sorted mode uses a stable permutation of
   execution indices and reduces each owner group before storing the result.
   The unsorted mode atomically accumulates in the coefficient buffers.
3. Fold the two or three local tiles onto both channel tori and expand the
   owner matrices with matrix products across all taps for each bank.
   Backward projects the Conv2d weight cotangent into the same basis and
   contracts the Atom coefficient derivatives. The signed aliases also
   apply to the radius and center derivatives.

The static mode processes the full declared bank set and supports replay
with prepared physical rows. The eager mode reads a live occupancy bitmask
and allocates/executes only banks occupied in that forward. Each autograd
invocation retains its own mask, geometry and optional owner order for its
backward. The rational decoder and optimizer were still evaluated eagerly.

## Measurement conditions

The matched RTX 3090 fixtures use FP32 channels-last input
`[256, 96, 8, 8]`, 128 output channels, stride 2, and 1,773 Atoms. Narrow
initial physical radii were about 1.44–6.71, wide ones 11.32–30.68. These are
channel radii. The 1×1 and 3×3 spatial windows each use the same input
and weight layout across methods. Each step includes rational radius
decoding, assembly, the weight layout conversion, convolution, backward,
and projected fused AdamW at learning rate 0.001 and zero decay. Dense uses
ordinary fused AdamW with the corresponding initial represented weights.
Each process used 8 warmups and 24 synchronized timing samples per arm,
with the order reversed between the two seeds. TF32 was disabled.

The final comparison used processes 4999 and 5014 on compute capability 8.6
with PyTorch 2.6.0+cu126, CUDA 12.6, cuDNN 9.5.1 and driver 580.95.05. The
table shows medians from CUDA events, in ms, across the two processes.

| Fixture | Dense | Direct | All Fourier, static | All Fourier, occupied banks | Occupied banks, sorted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×1 narrow | 0.289–0.370 | 1.297–1.312 | 1.734 | 1.550–1.852 | 1.589–1.950 |
| 1×1 wide | 0.268–0.311 | 2.425–2.596 | 1.740–2.003 | 1.510–1.788 | 1.538–1.801 |
| 3×3 narrow | 1.009–1.015 | 1.591–1.631 | 2.332–2.464 | 1.909–1.925 | 1.955–2.165 |
| 3×3 wide | 0.976–1.079 | 2.791–2.823 | 2.293–2.311 | 1.821–1.828 | 1.854–1.871 |

For 3×3 wide, occupied-bank selection reduced static Fourier peak
allocation from 60.69–61.49 to 42.53–42.57 MiB, with 80 and 66 MiB
reserved, respectively. Dense peaked at 43.01–43.05 MiB allocated and
64 MiB reserved. For 3×3 narrow the occupied-bank Fourier path peaked
at 54.19–54.99 MiB allocated and 64 MiB reserved. These are total
PyTorch allocator peaks. GPU process memory was unmeasured.

The best measured 3×3 route depends on the radius distribution. Direct
won in the narrow fixture and occupied-bank Fourier in the wide fixture.
The latter improved over the per-tap Fourier level-4 route at 3.141–3.144 ms.
For 1×1 wide, the per-tap level-5 hybrid measured 1.487–1.757 ms, close to
the occupied-bank Fourier result. Sorting added time to the occupied-bank
path on both draws in all four fixtures. The comparison covers synthetic
centers and radii. An end-to-end use of this result requires trained Atom
distributions and model-level timing. Dense still had the shortest step
in all four fixtures.

## Validation and evidence

The final extension passed 41 targeted tests, covering cutoffs, both
channel sizes, whole-cell quadrature, torus center shifts, all first-order
gradients, independent eager masks and static CUDA Graph replay. On the
timing fixtures all Fourier modes agreed with CPU FP64 for W, Y, dX,
amplitude, radius fraction and both center gradients, with maximum
relative L2 error 1.85e-6. The initial static Fourier experiment also ran
the two existing 2048-wide Linear CUDA tests. The final CPU suite recorded
2,219 passed, 2,627 skipped. Wheel and sdist builds passed.

The measured final source is `17c8a60248200b60018fb07172c74a5ad76ab8b7`.
The archive, driver, logs and receipts are preserved in the worktree's
ignored `output/conv-active-fourier-20261124/` and in
`cst-conv-channel-20261124/17c8a602/` relative to Kyutech home. The
initial static experiment is in `output/conv-all-fourier-20261124/` and
`cst-conv-channel-20261124/52d3607a/`, measured from
`52d3607adc9e5b2a0430cff9bd2a1279d0245b78`. For reproduction, extract
the sources into `source/` and extract the matching driver. Configure
Ninja, CUDA headers and the isolated extension cache as in the
[previous CUDA experiment](radius-channel-native-rtx3090-20261124.md).

```sh
export PYTHONPATH=source/src:source:source/tests TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=2
python -m pytest -q test_radius_channel_weight_cuda.py
python measure.py --seed 17 --source-commit 17c8a60248200b60018fb07172c74a5ad76ab8b7 --output run-17.json
python measure.py --seed 23 --source-commit 17c8a60248200b60018fb07172c74a5ad76ab8b7 --output run-23.json
```

| Final run file | SHA-256 |
| --- | --- |
| `source.tar.gz` | `ab8ca171e26540023df0c72a8b2759952ed1406952dbba826622783da34eb1e9` |
| `driver.tar.gz` | `42fa54623467f650d33f6a06ae7f160bb16cca14cc4c1e1c5ba72440b4a4704c` |
| `measure.py` | `2f6f2c4ad2cd0df9797981038b08272bb1fdea02bd97ccea79e2c8669dcd050d` |
| `run-17.json` | `68ef999a55d1a328185a7b1e7fbcb8486577ee73358fcac90940f7546b985dbb` |
| `run-23.json` | `3aadd857e3c140332afbb3b9653f7ee8e4dc1e854930729f2a7b1b1aaaf9e345` |
