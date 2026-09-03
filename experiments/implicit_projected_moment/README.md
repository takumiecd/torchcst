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
