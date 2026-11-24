# Fourier dispatch through CSTConv2d

**English** | [日本語](conv-fourier-dispatch-20261124.ja.md)

The `fourier` and `fourier_active` aliases force registered Plans through
the shared Dispatcher. A channel-tap layer binds one group of Atom IDs per
spatial tap and gathers them in ID order for execution. Checkpointing and
physical relayout preserve the mapping. Torch composition evaluates the same
mapping. The prior [kernel timing study](radius-channel-fourier-wideresnet-20261124.md)
used the private weight entry and remains tied to that source.

The RTX 3090 (compute capability 8.6), PyTorch 2.6.0+cu126 and CUDA 12.6
suite passed 87 tests. The public layer was checked against FP64 Torch on all
nine WideResNet shapes, including 3×3 stride-2 asymmetric padding, both Fourier
Plans and static Graph replay after changing radius fractions and centers.
The CPU suite passed 2,222 tests with 2,650 skips.

A separate smoke run replaced all 21 target convolutions in the WideResNet-22-2
topology from cst-research `6529bc0c8e8c8cb89816016a4579cfe958eecfa4`. It
used four Atoms per tap, batch 2 of synthetic inputs, a dense stem and
classifier, and one eager momentum-SGD step through CSTOptimizer. All 21
Atom tables received finite gradients and changed on the step. After the
step, the evaluation logits agreed for both Fourier Plans and Torch composition
in this fixture. This run did not measure model timing or accuracy.

The tested torchcst source is `fafae9033a3cfff62a8a1f2faadb64feb69be3b1`.
Source, scripts and logs are preserved in the worktree's ignored
`output/conv-fourier-dispatch-20261124/` and in the Kyutech home-relative
`cst-conv-channel-20261124/fafae903/`. On the matching runtime, extract the
archive into `source/` in a fresh experiment directory and repeat the checks with
`bash run-model.sh test` and `bash run-model.sh wideresnet`.

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `232498b24ffb98427aebdafc8bb07ef8bdf078e7eb14f16e22183a297f2b22ec` |
| `test.log` | `1c1249b4f602eaa4f9bbf54237b87db2606cdc3e861e79c04a70bf2d7d39c79c` |
| `run-model.sh` | `d78d60a4ad352e1b5215a0b3fecf4c219124f6ffb5a78c502478ca7aed2e1c05` |
| `verify_wideresnet.py` | `ea9a723aba6b46a09537e98a813364d6114e1c9acdffaf7395de8c8193ba6129` |
| `rigl_wide_resnet.py` | `a64ac48628c1a94f1e7f391cfed3b6dc1c9829fc61ecc13c0b90f4b336edaca6` |
| `wideresnet.json` | `dba9527baefd63d2622939efd8de69c87ab0dfb7f4167e4b61327986e8a35a31` |
