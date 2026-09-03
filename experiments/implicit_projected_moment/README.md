# Implicit projected-moment experiments

## Rank-one square-root D-side oracle

`rank_one_d.py` checks the tractable alternative to the elementwise Adam-like
signal

```text
D_diag = Diag(abs(vec(grad_W)) + eps).
```

For `g = vec(grad_W)`, the alternative is

```text
D_rank = sqrt(g g.T) + eps I = g g.T / ||g|| + eps I.
```

The square root is closed-form because `g g.T` has rank one.  Given the
dense-free P2 statistics

```text
b = J.T g
C = g contract H,
```

the complete candidate-dependent D-side force is

```text
r(d) = b + C d
s(d) = b.T d + 0.5 d.T C d
B_rank(d) = s(d) r(d) / ||g|| + eps V(d).T DeltaW(d).
```

The experiment verifies that formula against an explicit ambient oracle.  It
also verifies the compact identities for `J.T D J`, `J.T D H`, and `H.T D H`,
computes `||grad_W||` from batch-sized input/output-gradient Gram matrices, and
measures how far the rank-one force is from `D_diag` on random trust-region
directions.

Run the CPU smoke sweep:

```bash
PYTHONPATH=src python -m \
  experiments.implicit_projected_moment.rank_one_d \
  --preset smoke \
  --device cpu \
  --dtype float64 \
  --output output/implicit_rank_one_d_smoke.json
```

The experiment intentionally materializes the tiny represented weight,
Jacobian, Hessian, and D matrices.  They are correctness oracles, not a
production implementation.  The P2 `(b, C)` path and the batch-Gram gradient
norm are evaluated independently to show which inputs a later dense-free
implementation actually needs.

## EMA transport fidelity oracle

`ema_transport_fidelity.py` isolates the question of whether repeated
recompression loses information that becomes visible in later CST tangent
spaces.  It does not solve the quartic step problem.  Instead, every carrier
follows the same prescribed first-order trust-radius trajectory.

The tiny oracle retains never-compressed ambient first- and D-side EMA vectors.
It compares their pullbacks with a persistent compact state containing only
`(alpha, gamma, old_theta, old_step)`.  A base-tangent carrier is included as a
control for the document's accepted-point `V(d)` carrier.

Run the CPU smoke sweep:

```bash
PYTHONPATH=src python -m \
  experiments.implicit_projected_moment.ema_transport_fidelity \
  --preset smoke \
  --device cpu \
  --dtype float64 \
  --output output/implicit_projected_ema_transport_smoke.json
```

The important metrics are the relative errors of the current accepted-step
numerators `A_t(d_t)` and `G_t(d_t)` against the never-compressed dense EMA.
The zero-history control (`beta1 = beta2 = 0`) must agree to numerical
precision; nonzero error with EMA then measures cross-time projection loss,
not an error in the current contraction or the step solver.
