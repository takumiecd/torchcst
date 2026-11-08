# Radius-envelope implementation validation

English (canonical) | [日本語](radius-envelope-validation-20261108.ja.md)

The opt-in `kernels.radius_envelope` implementation passed the checks below.
[API and equations](radius-envelope.md), [compact verification](radius-envelope-validation-20261108.json).
This validates operator/optimizer behavior, not LM accuracy or an optimal tau.

## What ran

- Local full CPU suite:2433 passed/2408 skipped on `a86db78dc4fab6ad35c782d619a46b7bd3bfdcd6`.
  GPU/live skips do not count as validation. The final change only expands CUDA
  test cases: targeted local CPU24 passed/8 CUDA skipped on the final source.
- Final RTX6000 Ada job:40 tests passed, including **14 actual CUDA cases** and
  26 CPU/metadata cases. No skipped tests. The envelope matrix covers both
  512->2048 and2048->512,8192 Atoms, noncontiguous inputs with7/256 rows,
  radius maxima4.7/50, and a mixture of q0/q0.25/q1 and zero amplitudes.
- Physical rows from the CPU/GPU envelope maps agree exactly in FP32. The
  reference retains that encoding, then computes physical factors and sums in
  FP64. Y, dX and all four Atom fields pass atol1e-4/rtol4e-3; tolerances were
  not relaxed. Repeated backward agrees exactly. Actual torch.save/load of
  model and optimizer preserves the next Parameters, moments and counters.
- CPU checks include independent polynomial Gauss quadrature for Y/dX and
  finite differences for every Atom field, analytic a/q derivatives at both
  q endpoints, tau0.01/0.03/0.1, zero amplitude, extreme finite tau, encoded
  radius bounds, q learning/field LR, projection, fixed-constant precision,
  moment initialization and explicit Plan isolation.
- Existing physical FFN regressions pass both directions at7/65/256 rows.
  Original physical/Polar Plans reject the new stored-q contract.

The CUDA implementation still uses the existing physical arithmetic, with an
FP64 Python envelope map and FP32 physical rows. Default selectors and original
Polar update/gradient rules were not changed.

## Reference correction and preserved attempts

| Attempt suffix | Outcome | Interpretation |
| --- | --- | --- |
| 01 | 25 passed/8 failed | System environment lacked Ninja; CUDA build/import failed. |
| 02 | 25 passed/8 failed | CUDA header search path lacked cusparse.h. |
| 03 | 31 passed/2 failed | Double-converting latent CPU models changed encoded hierarchy branches. |
| 04 | 36 passed | Matched FP32 physical-row reference; full optimizer/resume checks. |
| 05 | 40 passed | Extended the same tests to radius_max50. |

At radius_max4.7, converting the FP32 latent model to double changed the
radius0.5 branch for470/8192 Atoms. Thus it compared different functions, rather
than only improving summation precision. With matching encoding, the diagnostic
maximum absolute differences across its two maps were Y5.54e-7, dX9.09e-7 and
Atom gradient2.86e-6. These diagnostic maxima are not claimed as maxima of the
entire final test matrix. The boundary behavior and reference are regression
tested. No runtime source changed after the original implementation commit
`ee9c89556e99b09ed993e3a1a8aff874e3a1ee99`; later changes corrected tests/docs.

## Provenance and reproduction

Actual NVIDIA RTX6000 Ada Generation,48GB class, CC8.9; Python3.13.11,
Torch2.9.1+cu128, CUDA12.8. Frozen final source
`40e120c9f1b7260364f08bf6723185f62bf31ac6`; source archive SHA256
`c9e11d23d94a8e83f1d257af35f380a58d78076ec6c46828f18940ff0d3aedcf`.
Job `radius-envelope-6000ada-20261108-05` used a private extension directory and
the previously verified venv with Ninja; NVIDIA-wheel include directories were
provided through CPATH. Exact environment/driver records remain in ignored
owner evidence, without copying machine-specific paths into this document.

```bash
python -m tools.kernel_dev test --suite cpu
python -m pytest tests/test_radius_envelope.py tests/test_lm_physical_radius.py -q
```

The second command needs the qualified CUDA GPU and a complete extension build
environment. Setting CPATH to the installed NVIDIA-wheel include directories
was needed on this host. No package/runtime was modified in place.

The compact archive SHA256 is
`99d6f25b5295642b011a60387345bb399ef4fd917c0aeee9d523e8af28a1c24e`.
Retrieval independently verified28 files, all1306 frozen source files per
attempt against the corresponding local Git archive, driver hashes, and the
unchanged runtime source. All five recorded controllers were inactive; GPU
postflight showed0% utilization/39MiB. Failed attempts, archives and native
builds remain preserved. Local evidence is under ignored
`output/radius-envelope-6000ada-20261108-01` through `-05` in the owner's
canonical torchcst checkout. This scope does not include LM training, a tau
sweep, speed/peak-memory comparisons, other GPU cohorts or CUDA Graph capture.
