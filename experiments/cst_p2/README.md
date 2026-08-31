# Dense-free P2 optimizer ablation

`dense_free_ablation.py` compares four update rules at an identical update
budget and trust-region radius:

- `p1_current`: current minibatch linear model (`C = 0`).
- `p2_current`: current minibatch P2 model.
- `p2_model_ema`: EMA of the complete `(b, C)` model before solving.
- `p2_step_ema`: current P2 model, then EMA of the solved dimensionless step.
- `p2_model_ema_adaptive`: model EMA plus minibatch loss-ratio acceptance and
  trust-region radius adaptation.

Both teacher and learner are evaluated in factored form. The experiment
monkeypatches `CSTLinear.dense_weight` to raise, so a materialized dense weight
is a test failure. `relative_mean_loss` is the mean full-dataset loss after
each update divided by the initial loss; it is the step-budget comparison
(lower is better). `relative_final_loss` measures the endpoint.

Run:

```bash
conda run -n project python -m experiments.cst_p2.dense_free_ablation \
  --device cpu --steps 60 --seeds 3
```

The optimizer retains `K(p + p²)` model-EMA elements. Step EMA, when enabled,
adds `Kp`; it never adds an `[out, in]` state tensor.

The first checked result and interpretation limits are in
[`baseline_results.md`](baseline_results.md).
