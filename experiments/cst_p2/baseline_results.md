# Dense-free P2 baseline results

## 2026-08-31: factored regression, 60 updates

Command:

```bash
conda run -n project python -m experiments.cst_p2.dense_free_ablation \
  --device cpu --steps 60 --seeds 3
```

The teacher and learner are factored Gaussian/L2 CST linear maps. The dataset
has 192 samples and a `16 -> 12` represented map. The learner has 10 atoms and
30 trainable scalar fields. `dense_weight()` is monkeypatched to raise.

| arm | final loss / initial | mean step loss / initial | update ms | extra step state |
| --- | ---: | ---: | ---: | ---: |
| P1, current model | 0.2611 +/- 0.0663 | 0.5558 +/- 0.0979 | 4.77 | 0 |
| P2, current model | 0.2585 +/- 0.0668 | 0.5538 +/- 0.0972 | 4.33 | 0 |
| P2, model EMA | 0.2469 +/- 0.0602 | 0.5475 +/- 0.0897 | 4.35 | 0 |
| P2, step EMA | 0.2782 +/- 0.0569 | 0.5664 +/- 0.0963 | 4.42 | 30 |
| P2, model EMA + adaptive radius | **0.0564 +/- 0.0476** | **0.1916 +/- 0.0490** | 4.45 | 0 |

All arms retain 120 model values: `K(p + p^2) = 10(3 + 9)`. The dense map
would contain 192 values in this intentionally tiny problem; the scaling claim
is about larger represented maps, where retained optimizer state remains tied
to atom count rather than `out_features * in_features`.

The adaptive arm accepted 45.33 and rejected 14.67 of 60 proposals on average,
ending at mean radius 0.28 from an initial 0.06. Relative to fixed-radius model
EMA, its endpoint ratio is 4.38 times smaller and its step-mean ratio is 2.86
times smaller.

## Interpretation boundary

This result supports the combination of a temporally averaged P2 model and a
trust test; it does not isolate P2 curvature as the sole cause. The adaptive arm
is allowed to grow its radius and performs one additional factored minibatch
forward for acceptance. Timing is therefore informative only for this small CPU
case, not yet a CUDA performance conclusion.

Acceptance is minibatch-local. Across the three runs, the full-dataset loss
still increased on 3.33 accepted-or-restored iterations on average, showing
that stochastic acceptance noise remains. A held-out or averaged ratio test is
a future ablation. MNIST accuracy, the weight-gated sigma controller, repulsion,
and an analytic/fused contraction are also not part of this baseline.
