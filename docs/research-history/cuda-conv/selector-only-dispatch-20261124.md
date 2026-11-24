# Selector-only layer dispatch

**English** | [日本語](selector-only-dispatch-20261124.ja.md)

CSTLinear and CSTConv2d take `selector=` as the execution configuration.
The Conv2d patch layout uses its previous size rule for factored and
materialized Plans. Grouped patch convolutions use the materialized Plan.
Channel taps default to Torch composition. The patch composition and two
Fourier Plans can be selected with a FixedSelector. All Conv2d Plans use
the shared Dispatcher.

The CPU suite passed 2,226 tests with 2,650 skips. The wheel and sdist built.
The targeted suite passed 126 tests on an RTX 3090 with PyTorch
2.6.0+cu126 and CUDA 12.6. It covered both Fourier Plans on all nine
WideResNet convolution shapes, CUDA Graph replay, Linear default selection,
and an explicitly selected wide rational Linear comparison.

The WideResNet-22-2 smoke run updated all 21 convolution Atom tables in one
synthetic batch of size 2 with momentum SGD through CSTOptimizer. After the
step, both fixed Fourier Plans produced the same evaluation logits as the
default Selector. The stem and classifier used ordinary Torch layers.
This is a single-step correctness check.

The tested source is `985ec3dcff677839c7dc172c940b74b89ffc83e5`. Evidence is
in the worktree's ignored `output/selector-only-dispatch-20261124/` and the
Kyutech home-relative `cst-conv-channel-20261124/985ec3dc/`. In a fresh
experiment directory on the matching runtime, extract the archive into
`source/`, then run `bash run.sh test` and `bash run.sh wideresnet`.

| File | SHA-256 |
| --- | --- |
| `source.tar.gz` | `6123d120bbf935e1f9d31a5aacad8225af47ea85daafe7f381dbbdec9b040c77` |
| `run.sh` | `dac21a3ceaf67a10498cd0e894da53093741457e0bce9db5b9a363ddff6f3cae` |
| `test.log` | `8869de207a8afdbfcc7caf673f0a5a5d3f021dfbd756ca911ee8b578eb37fe27` |
| `wideresnet.json` | `dba9527baefd63d2622939efd8de69c87ab0dfb7f4167e4b61327986e8a35a31` |
| `verify_wideresnet.py` | `d7cd94501515cb77234fedfed29c4acb728bc6fb37a2b724c48c51ceca262ba9` |
| `rigl_wide_resnet.py` | `a64ac48628c1a94f1e7f391cfed3b6dc1c9829fc61ecc13c0b90f4b336edaca6` |
