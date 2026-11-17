# RTX 4090 FP32 QKV variance calibration

English | [日本語](lm-qkv-gpu-initialization-20261117.ja.md)

The width-2048 TinyStories QKV study uses q=0 and an exponential lower
envelope during weight initialization. This research path retains the CPU
random stream and existing bounded scalar scale search. It computes each
candidate's physical radius and hierarchy bank on the CPU in FP64 before
transferring the physical rows and bank indices to CUDA. The FP32 kernel
computes the squared cell-mass norms from Fourier coefficients for periods
longer than 16 sites, with direct cell integration on shorter periods. The
physical grid has 2048 sites per axis and the largest local period is 1536.

The first all-FP32 experiment, lm-qkv-gpu-init-20261117-01, rounded some
radii above 0.5 down to 0.5 before selecting the bank. This changed their
local period and all 72 fitted parameter tensors. The revised experiment
selects the bank before rounding. It also reduces the center modulo each
short local period before directly evaluating squared cell masses.

| Atoms per projection | CPU calibration | GPU-assisted calibration | Sample energy relative L2 |
| ---: | ---: | ---: | ---: |
| 131,072 | 33.376 s | 31.824 ms | 2.178e-7 |
| 262,144 | 70.465 s | 59.923 ms | 2.531e-7 |

These synchronized wall-clock measurements cover the scale search for the
same block-zero Q projection at seed 17. The GPU path includes the CPU
radius and bank preparation and transfers. It is measured after a kernel
warm-up. Module construction, CUDA compilation, and LM training are outside
this timer. The paired CPU and GPU calibrations returned bitwise identical
Atom rows for both counts. The GPU result also matched the Atom rows
reconstructed from the study's recorded CPU scales and seeded tensors in
all 72 projections across seeds 17, 23, and 42. The largest relative
difference in fitted energy across those projections was 1.643e-7.

Use the archived lm-qkv-2048-20261117-01 torchcst snapshot and overlay the
candidate delta from 9c0cf9979dde545999b290f6e0ec13ff93ecf2b5 to
27ca1390e24fbb5f3f0c854b09c1fc0585472add. The study's verified summary is the
input. Set PYTHONPATH for the resulting snapshot, select a fresh
TORCH_EXTENSIONS_DIR, and build for TORCH_CUDA_ARCH_LIST=8.9 with PyTorch
2.9.1+cu128. Use the research benchmark on the RTX 4090.

~~~sh
python -m benchmarks.cuda.linear.lm_qkv_initialization \
  --study "$TRAINING_EVIDENCE/summary.json" --output "$CALIBRATION_RESULT"
~~~

The verified result is lm-qkv-gpu-init-20261117-02. Its SHA-256 is
be0e4c189193ed67c4ebe22e9261e926ae974f05ee924f352e06c3b35363e78a.
The training summary SHA-256 is
15ce5cc578b817fe94b39e0dfd84ac268520d6a36a615c2e4c23e8ff23fbceda.
