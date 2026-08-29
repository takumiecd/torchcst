# CST curvature decomposition

This experiment asks whether ordinary loss curvature in materialized weight
space can remain ignored while retaining the CST-specific curvature of the
parameter-to-weight map.

For one Gaussian CST layer, let `theta` contain each atom's `(w, s, t)` and
let `W(theta)` be its materialized weight.  One plain SGD step is evaluated
with four loss-change predictions:

| Name | `theta -> W` | `W -> loss` |
| --- | --- | --- |
| `p1` | first order | first order |
| `p2_cst` | second order | first order |
| `p_weight_exact` | exact finite map | first order |
| `p_actual` | exact finite map | actual loss |

The loss is MSE, so it is exactly quadratic in `W`.  Therefore
`p_actual - p_weight_exact` is exactly the omitted weight-space curvature for
the actual CST update; it contains no third- or higher-order loss remainder.
The script also reports the quadratic `J.T H_W J` contribution as `HW2(Jd)`.
Within the atom-local CST map Hessian, `Hwp2` is the amplitude-position
off-diagonal contribution and `Hpp2` is the position-position contribution
along the actual SGD step.

Run the default one/two-atom amplitude and learning-rate sweep from the
repository root:

```bash
PYTHONPATH=src python -m experiments.cst_curvature.decomposition
```

Run a focused near-zero-amplitude sweep:

```bash
PYTHONPATH=src python -m experiments.cst_curvature.decomposition \
  --atoms 1 2 \
  --amplitudes 0 1e-10 1e-8 1e-6 1e-4 \
  --learning-rates 1e-3 1e-2 1e-1
```

Interpret the gaps in order:

1. `p1 -> p2_cst`: CST map second-order effect.
2. `p2_cst -> p_weight_exact`: CST map third and higher orders.
3. `p_weight_exact -> p_actual`: curvature of the ordinary loss in `W`.

`cross_map_H` is the maximum absolute cross-atom entry in the contracted CST
map Hessian.  It should be zero because `W(theta)` is additive over atoms.
At exactly zero amplitude, ordinary SGD also has a zero position step, so the
amplitude-position Hessian exists but its directional contribution is zero.
Near-zero rather than exactly-zero amplitudes reveal whether that block changes
the predicted loss decrease under the actual SGD direction.
