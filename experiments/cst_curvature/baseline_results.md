# Baseline direction results

These are the first deterministic results from `decomposition.py`.  They are
not a general CST conclusion: the model has one Gaussian CST layer, 5 input
neurons, 4 output neurons, batch size 8, one-dimensional charts, MSE, float64,
and seed 17.  Every direction except the legacy raw SGD step has radius `0.01`
in

```text
q = (w / 1.0, s / sigma, t / sigma),  sigma = 0.28.
```

The table reports

```text
abs(CST-map quadratic) / abs(J.T H_W J quadratic).
```

`random` is the median of 32 seeded directions.

| atoms | amplitude | SGD unit | random | inverse pullback | bounded pullback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1e-6 | 6.86e-6 | 3.10e-1 | 3.67e5 | 1.52 |
| 1 | 1e-3 | 6.79e-3 | 3.10e-1 | 3.67e2 | 1.55 |
| 1 | 1e-1 | 3.69e-1 | 3.59e-1 | 3.48 | 1.74 |
| 2 | 1e-6 | 4.12e-6 | 2.20e-1 | 2.70e5 | 7.25e-1 |
| 2 | 1e-3 | 4.01e-3 | 2.20e-1 | 2.70e2 | 7.43e-1 |
| 2 | 1e-1 | 1.08e-1 | 1.93e-1 | 1.47 | 7.74e-1 |

The near-zero inverse-pullback orientation is qualitatively different from
SGD.  At one atom and `w=1e-3`, the normalized steps are approximately

```text
SGD unit:          |delta_w| = 9.998e-3, |delta_position| = 6.027e-5
inverse pullback:  |delta_w| = 5.054e-6, |delta_position| = 2.800e-3
bounded pullback:  |delta_w| = 1.486e-3, |delta_position| = 2.769e-3
```

Inverse pullback makes the position displacement finite while its amplitude
displacement scales down with `w`.  Consequently the ordinary loss-curvature
term becomes quadratic in the small delivered weight change, while the CST
map curvature retains its amplitude-position and position-position effects.
For one atom at `w=1e-3`, adding only CST map curvature reduces relative loss
change prediction error from `9.77e-3` to `1.14e-5`; the omitted `H_W` term is
about 367 times smaller than the CST map quadratic.

Bounded pullback does not suppress the amplitude displacement as strongly.
Its CST and ordinary loss curvature remain comparable in these trials.  Over
unstructured dimensionless random directions the ordinary loss curvature is
usually larger, although the CST term is not zero (median ratio roughly
`0.2--0.36`).

The contracted CST map Hessian's cross-atom entries were zero in every trial,
as required by the additive atom map.  This does not force the full loss
Hessian's cross-atom entries to zero; those belong to the separately measured
`J.T H_W J` term.
