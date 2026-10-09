# Exact Torus Grid-window negative result and recovery

The rejected route used the actual rounded S² site snapshots, not an ideal unrounded grid. With q=Q a, s=n b, actual site norm bounds l≤n≤u and an outward-inflated chord radius d, positive raw support implies

\[
\|s-q\|^2=n^2+Q^2-2nQ(a\cdot b)\le d^2,
\qquad a\cdot b\ge\frac{l^2+Q^2-d^2}{2Qu}.
\]

For a northern cap with h=√(1−t²), the gnomonic coordinate extrema are

\[
r\,\frac{a_0a_k\pm h\sqrt{a_0^2+a_k^2-h^2}}{a_0^2-h^2}.
\]

The implementation expanded these by actual-site/ideal-grid projection discrepancy and outward rounding, falling back to complete traversal for unsafe/nonfinite/horizon or broad windows. It preserved the original FP32 final support classifier, coupled q0 derivative, one combined normalization floor and exact overflow. All direct positive-support witnesses and physical gradients passed. This mathematical contract and detailed rounding proof remain recoverable at measured branch `kernel/torus-grid-window-measured-bdea` (bdeaed973) and final negative branch `kernel/torus-grid-window-negative-final` (1e9e0c4).

Full16worker primary2bf1 on L4 found window32/window16 slower in every fixed N1024/2048 sigma3/8 case by36–67%. Int16 retained17.665%/20.992% memory savings, but the speed regression failed the joint gate; inverse selection was empty. Forward time added about7.8ms/30ms for sigma3; backward/optimizer stayed stable. Per-atom FP64 bound/control-flow/occupancy cost is a hypothesis, not isolated evidence.

Original failed common gate7da, corrected common1cb, failed diagnostic capture21d9, corrected E2Ed14, catalog typo61e721 and transport-interrupted5552 remain preserved separately from successful primary2bf1. No partial/interrupted comparison was selected, no tolerance or case was removed. Root restore/fsck and fullraw SHA verification preceded the new branch. Recovery bundle/raw archive lives under the owner workspace `output/torus-compact-ids-20261009/archives/window-negative`; this note does not add the rejected runtime to main.

Recovery Git bundle SHA256 is
`044121ee121f59008602436bd9925789012f2f3401f67bd5d31758e34bb47220`;
manifest records restore/fsck verification and164raw-file copied hashes. The
negative primary2bf1 source/result archive hashes are
`0881f89d2fbeb82bb0bf0d7d6c9bdbe5452ebd42f3a0d5b9adf62e2d46df5a40` /
`9bc9ec2574ae9971ccde4f57fe75b8d596016b7954b2b7c9bf3f82446e54113e`.
Actual L4 UUID39896750-9b1b-bf81-b223-d3d615bec420,driver580.82.07,
Torch2.11.0+cu130,Triton3.6.0. The later compact-ID full-scan study is a separate
candidate retaining the original prep, not a reclassification of this failure.
