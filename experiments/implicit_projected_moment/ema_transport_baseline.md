# Projected EMA transport smoke result

This experiment measures the intended question directly: after every accepted
step, can `(alpha, gamma, old_theta, old_step)` preserve the parts of a dense
EMA that become visible through later CST tangent spaces?

It does not solve or approximate the quartic step problem.  All carriers follow
the same first-order, normalized trust-radius trajectory, so the only measured
difference is repeated state projection.  The dense EMA exists only inside the
small correctness oracle and is never projected.

The CPU/float64 smoke sweep contains 24 trials: one/two atoms, radii `1e-3`,
`1e-2`, and `5e-2`, two seeds, 24 steps, and matched zero-history controls.
The moving-history trials use `beta1 = 0.9` and `beta2 = 0.99`.  The D-side
sample uses

```text
D_t DeltaW_t = g_t (g_t.T DeltaW_t) / ||g_t|| + eps DeltaW_t.
```

## Result

For the document's accepted-point `V_t(d_t)` carrier, relative to the
never-compressed dense EMA:

- median current `A_t(d_t)` relative error: `0.00319` (`0.319%`);
- maximum current `A_t(d_t)` relative error: `0.02205` (`2.205%`);
- median current `G_t(d_t)` relative error: `0.00438` (`0.438%`);
- maximum current `G_t(d_t)` relative error: `0.03241` (`3.241%`);
- minimum first-side cosine: `0.999883`;
- minimum D-side cosine: `0.999700`.

Error scales with how far the visible space moves.  At radius `1e-3`, median
first/D-side errors were `0.0236%` and `0.0308%`.  At radius `1e-2`, they were
`0.695%` and `0.969%`.  At radius `5e-2`, they were `1.048%` and `1.243%`.

Compressing into the accepted-point `V_t(d_t)` was consistently better than
compressing into the base `J_t`.  Across the sweep, the base carrier's median
first/D-side errors were `0.364%` and `0.465%`, versus `0.319%` and `0.438%`
for the accepted carrier.  The improvement is modest in this smooth toy map,
but it has the expected sign at every tested radius.

## Controls and interpretation

With `beta1 = beta2 = 0`, both current numerators matched the dense reference
exactly.  Expand-add-recompress preserved its own current observable numerator
within `1.42e-14`.  The nonzero moving-history error therefore measures only
information discarded by earlier projections and exposed by later visible
spaces.

This is positive evidence that compact projected EMA can retain the
optimizer-visible dense history accurately on smooth, small-step Gaussian CST
trajectories.  It is not yet a claim about final task accuracy, large atom
counts, structural mutations, long runs, or abrupt visible-space rotation.
Those are the next stress axes; the quartic solver can remain completely
separate from this fidelity experiment.
