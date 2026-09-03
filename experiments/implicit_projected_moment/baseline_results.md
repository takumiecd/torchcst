# Rank-one D-side smoke result

The first CPU/float64 smoke sweep ran 32 Gaussian CST problems over one/two
atoms, amplitudes `1e-3`/`1e-1`, residual scales `1e-3`/`1e-1`, coincident and
separated atoms, and two seeds.  Each problem evaluated 16 random
dimensionless trust-region directions.

## Mechanism result

The rank-one square-root identity is numerically sound.  With

```text
g = vec(grad_W)
D_rank = sqrt(g g.T) + eps I
b = J.T g
C = g contract H,
```

the compact force

```text
r(d) = b + C d
s(d) = b.T d + 0.5 d.T C d
B_rank(d) = s(d) r(d) / ||g|| + eps V(d).T DeltaW(d)
```

matched the explicit ambient oracle with maximum relative error `7.57e-14`.
The corresponding energy matched within `1.12e-12`.  The independent
dense-free P2 reconstruction of `b` and `C`, and the batch-Gram reconstruction
of `||grad_W||`, all matched within `1.31e-15`.  The compact identities for
`J.T D J`, `J.T D H`, and `H.T D H` also matched within `8.94e-15`.

The undamped pulled-back rank-one operator had effective rank exactly one in
every trial, as required.

## Comparison with elementwise `Diag(abs(grad_W) + eps)`

The rank-one square root is tractable, but this smoke result does not support
treating it as a drop-in numerical approximation to the elementwise diagonal:

- median force cosine over all trials: `0.9965`;
- minimum per-trial median force cosine: `0.5217`;
- median force relative error: `0.9141`;
- median rank-one/diagonal energy ratio: `0.0830`.

The high overall cosine hides a harder regime.  For amplitude `1e-1`, the
median force cosine was only `0.7132`; for two atoms it was `0.9282`.  The raw
rank-one energy was also roughly twelve times smaller at the overall median,
so learning-rate or trace calibration would be required even where the force
orientation agrees.

For two atoms, the median cross-atom fractions of the linearized pullback were
`0.5874` for the elementwise diagonal and `0.6936` for the rank-one square
root.  A same-atom approximation is therefore a material, separately measured
approximation in this setup.

## Decision

`sqrt(g g.T)` is a viable dense-free D-side research arm because it needs only
the existing P2 `(b, C)` statistics and a batch-sized calculation of `||g||`.
It should not yet replace the elementwise diagonal arm.  The next experiment
should compare actual quartic trust-region steps and loss reduction after
calibrating their overall D scale; force-vector similarity alone is not enough
to select an optimizer.

The complete smoke output is written by the runner to
`output/implicit_rank_one_d_smoke.json`.
