# Dense-free P2 optimizer

## Research contract

The candidate optimizer must preserve CST's memory advantage. Its training
path therefore must not materialize or retain any tensor shaped like the dense
represented weight:

- no dense `W`;
- no dense `grad_W`;
- no dense `m_W` or `v_W`;
- no materialized map Jacobian or map Hessian.

An explicit `Factored` backend is required while the first implementation is
being validated. `"auto"` is not sufficient because a live-atom crossover may
silently select dense materialization. Small dense calculations remain lawful
only in tests as correctness oracles and as external accuracy baselines.

For `K` atoms with `p` fields per atom, persistent optimizer state should be
`O(K p)` and the same-atom P2 curvature may use `O(K p^2)`.

## Sufficient statistics

Freeze the output-side loss gradient and write the additive CST map as

```text
W(theta) = sum_k W_k(theta_k).
```

For one atom define

```text
score_k(theta_k) = <stopgrad(grad_W L), W_k(theta_k)>.
```

The dense tensor in that notation is conceptual. For a linear layer and one
captured `(x, grad_output)` pair, the identical scalar is computed without it:

```text
score_k = w_k * sum_batch
          <x, k_in(theta_k)> * <grad_output, k_out(theta_k)>.
```

Its first two derivatives are

```text
b_k = J_k^T grad_W L
C_k = grad_W L contract H_map,k.
```

Consequently the P2 loss model is

```text
q(d) = b^T d + 1/2 d^T C d.
```

No information from `grad_W` that can affect this P2 subproblem is discarded:
all such information is present in `(b, C)`. Because the represented map is a
sum over atoms, `C` is block diagonal across atoms even though the ordinary
loss Hessian need not be.

## Temporal averaging

The first stochastic optimizer candidate averages the complete batch-local P2
model with one decay:

```text
bar_b_t = beta * bar_b_(t-1) + (1-beta) * b_t
bar_C_t = beta * bar_C_(t-1) + (1-beta) * C_t.
```

This is model EMA, not Adam. Adam's second moment is `EMA(b_t**2)`, which is
non-negative gradient-scale history; `C_t` is signed map curvature. An Adam-like
second moment is a later independent ablation, not a replacement for `C_t`.

The averaged step solves

```text
min_d  bar_b^T d + 1/2 d^T bar_C d
s.t.   ||d / scales|| <= radius.
```

The current solver diagonalizes each same-atom block, then finds the one shared
KKT multiplier by scalar bisection. It also handles the negative-curvature hard
case. Its storage is `O(K p^2)`; it does not assemble the global `Kp x Kp`
matrix.

## Acceptance and radius adaptation

The optional adaptive path evaluates the proposed atom update on the same
minibatch and compares actual reduction with the P2 prediction:

```text
r_t = (L_before - L_after) / (-q(d)).
```

An update is accepted only when its actual reduction is positive and `r_t`
exceeds the configured acceptance threshold. A rejected proposal restores only
the `(w, s, t)` atom tensors, so rollback is `O(K p)` and never needs a dense
weight snapshot. The trust radius is halved after a rejection or a poor ratio;
a boundary step with a strong ratio doubles it up to `max_radius`.

This is deliberately separate from model EMA. EMA reduces minibatch noise in
the estimated P2 model; the ratio test controls how far that model may be
trusted. Step EMA was also implemented as a separate ablation in normalized
trust-region coordinates, but was inferior in the first comparison.

## Minimal training loop

`CSTP2TrustRegion` owns the live atom amplitudes and coordinates. One update is
captured and applied as follows:

```python
optimizer = CSTP2TrustRegion(
    layer,
    radius=0.1,
    beta=0.9,
    amplitude_scale=1.0,
    adaptive_radius=True,
)

context = optimizer.capture_context(update_id)
layer.set_backward_context(context)
optimizer.zero_grad()
loss = loss_fn(layer(inputs), targets)
loss.backward()
context.observe_microbatch()
capture = context.finalize_capture()
layer.set_backward_context(None)
optimizer.step(
    capture,
    closure=lambda: loss_fn(layer(inputs), targets),
    current_loss=loss.detach(),
)
```

The closure is evaluated under `torch.no_grad()` and must use the same captured
minibatch when its ratio is intended to validate that minibatch model. It adds
one factored forward evaluation, not a dense materialization.

Multiple microbatches may be queued into the same context. Their already
loss-scaled `(b, C)` models are combined using `micro_weight`, then exactly one
model enters the temporal EMA. Raw activations and output gradients are dropped
inside the backward hook; the finalized capture retains only `K p + K p^2`
values per microbatch.

The optimizer checkpoint contains its model EMA and trust-region configuration.
A structural mutation invalidates the moving frame and is rejected until
`reset_model_ema()` explicitly clears the old history.

## Current implementation boundary

The first checked path is `CSTLinear` with `(w, s, t)` atoms and an explicit
`Factored` backend. Both the ordinary amplitude gauge and L2-normalized columns
can be differentiated. Structural mutation currently requires an explicit EMA
reset. Factor-declared atom columns, automatic lifecycle-following state,
convolution, the weight-gated bandwidth controller, repulsion, exact-loss
acceptance on a less noisy validation statistic, and a fused CUDA kernel remain
future integration work. Same-minibatch acceptance and radius adaptation are
implemented now.

The dense-free contractions are checked against dense-weight derivatives only
inside `tests/torchcst/test_p2_optimizer.py`. The same tests enforce rejection
of a non-explicit-factored backend and verify the trust-region KKT conditions.
The end-to-end factored test monkeypatches `dense_weight()` to raise, runs 50
updates, checks a greater-than-10% MSE reduction, and accounts for every retained
optimizer-state element.

The reproducible first ablation and its limitations are recorded in
`experiments/cst_p2/baseline_results.md`.
