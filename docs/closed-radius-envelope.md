# Closed radius envelopes (research opt-in)

English (canonical) | [日本語](closed-radius-envelope.ja.md)

This experimental Polar radius map joins both bounds at zero amplitude and at
the amplitude limit. It changes the radius contract, not the Polar activity
update or the detached-radius task derivative. It is not a recommended default.

For `x = abs(a) / amplitude_max`, `b = clamp(1-x*x, 0, 1)`:

$$L=r_{min}+(r_{max}-r_{min})b^{p_L},\qquad
U=r_{min}+(r_{max}-r_{min})b^{p_U},\quad p_L\ge p_U\ge1.$$

The radius remains `exp((1-q)*log(L)+q*log(U))`, clamped to `[L,U]` for rounding.
Both bounds are `r_max` at zero and `r_min` at the amplitude limit. Between
these endpoints q selects a radius; at the endpoints it has no radius authority.
Increasing either power makes its bound fall sooner. This alone is not evidence
that the exploration range, initial q or learning behavior improves.

```python
coordinates = kernels.polar(
    amplitude_max=1.0, w_c=0.3,
    bounds=RadiusBounds(minimum=0.5, birth=50.0, maximum=50.0, upper_floor=0.5),
    radius_envelope="closed_polynomial",
    closed_lower_power=4.0, closed_upper_power=1.0,
)
```

Use `RadiusBounds`, `birth=maximum`, and `upper_floor=minimum` on both axes.
The closed law does not use the rational law's w_c/kappa/decay controls; they
remain declared for the unchanged Polar representation/update. The default
`radius_envelope="rational"` retains the prior radius formula, buffer layout
and optimizer checkpoint manifest.
Closed powers are fixed checkpoint buffers; the envelope kind is checkpoint
contract metadata, and loading across envelope kinds is rejected.

Initialize on CPU using `init.variance_scaling_` before moving to CUDA. Radius
changes require recalibrating the amplitude scale to the same Dense second
moment, with the returned error/clipping report retained. A bounded search may
reject an unreachable target; do not train an unmatched fallback silently.

Native closed-envelope support is limited to the explicit
`GLOBAL_ACTIVE_GRAD4_NATIVE` and `GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048`
schedules. Other native schedules reject this declaration. Qualify Y, dX, all
Atom gradients and optimizer/checkpoint behavior for the actual workload before
training. CPU mathematical checks are not GPU or learning evidence.

## Gradual opening at zero

A separate opt-in, `radius_envelope="rational_zero_opening"`, retains the
existing rational upper envelope U0 and blends its lower envelope L0:

$$g(a)=\frac{a^2}{a^2+\tau^2},\qquad L=U_0-g(a)(U_0-L_0),\qquad U=U_0.$$

Pass `zero_opening_scale=tau` (positive, amplitude units). At zero L=U;
at |a|=tau the relative gap is one half; farther away the original interval
is recovered gradually. This law does not impose closure at amplitude_max.
For example use the preceding declaration with `birth=4`, `upper_floor=1`,
`radius_envelope="rational_zero_opening", zero_opening_scale=0.03`, omitting
closed powers. Tau is a fixed saved scalar, not a trainable q coordinate.
CPU/CUDA use a stable equivalent blend and endpoint interpolation. The same
native schedule restrictions, variance calibration and qualification rules
apply. Existing rational and closed-polynomial checkpoint manifests are preserved.
