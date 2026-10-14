"""Independent NumPy FP64 actual-site image-sum oracle, no shared moments.

Every atom/site value and centre derivative is evaluated independently before
the actual discrete product norm and one global floor. No runtime profile
function, coefficient, width table, Gram identity or prepared CUDA data is read.
"""

import numpy as np
import torch


def native_periodic(distance, sigma, period):
    """Peak-one image sum and its displacement derivative, direct closed form."""
    b = np.asarray(sigma, dtype=np.float64) ** 2
    d = np.remainder(np.asarray(distance, dtype=np.float64), period)
    length = period / b
    q = np.exp(-length)
    one_minus_q = -np.expm1(-length)
    a = 1 / one_minus_q
    c = length * q / one_minus_q**2
    r, z = d / b, (period - d) / b
    er, ez = np.exp(-r), np.exp(-z)
    raw = er * (a * (1 + r) + c) + ez * (a * (1 + z) + c)
    dd = (-er * (a * r + c) + ez * (a * z + c)) / b
    peak = a + c + q * (a * (1 + length) + c)
    return raw / peak, dd / peak


def native_mixture(distance, sigma, period, knots):
    """Interpolate native fixed widths at actual sites, derivative w.r.t. centre."""
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), knots[0], knots[-1])
    knots = np.asarray(knots, dtype=np.float64)
    index = np.clip(np.searchsorted(knots, sigma, side="right") - 1, 0, len(knots) - 2)
    weight = np.clip(
        (np.log(sigma) - np.log(knots[index]))
        / (np.log(knots[index + 1]) - np.log(knots[index])),
        0,
        1,
    )[:, None]
    lo, dl = native_periodic(distance, knots[index, None], period)
    hi, dh = native_periodic(distance, knots[index + 1, None], period)
    return (1 - weight) * lo + weight * hi, -((1 - weight) * dl + weight * dh)


def _polar(kernel, p):
    """NumPy declaration of Polar amplitude, live width, and amplitude VJP."""

    def scalar(name):
        return float(kernel.scalar(name).detach().cpu().double())

    radius2 = (p[:, :2] ** 2).sum(-1)
    safe_radius2 = np.maximum(radius2, np.finfo(np.float64).tiny)
    radius = np.sqrt(safe_radius2)
    amax = scalar("amplitude_max")
    amp = amax * p[:, 0] / radius
    alpha = np.clip((radius2 - 1) / 3, 0, 1)
    activity = (amp / scalar("w_c")) ** 2
    minimum = scalar("sigma_min_input")
    lower = minimum + (scalar("sigma_birth_input") - minimum) / (
        1 + scalar("lower_kappa") * activity
    )
    upper = minimum + (scalar("sigma_max_input") - minimum) * scalar("kappa") / (
        scalar("kappa") + activity ** scalar("upper_decay_power")
    )
    upper = np.maximum(np.maximum(upper, scalar("upper_floor_input")), lower)
    sigma = np.clip(
        np.exp((1 - alpha) * np.log(lower) + alpha * np.log(upper)), lower, upper
    )
    da = np.empty_like(p[:, :2])
    da[:, 0] = amax * (1 / radius - p[:, 0] ** 2 / radius**3)
    da[:, 1] = -amax * p[:, 0] * p[:, 1] / radius**3
    return amp, sigma, da


def oracle_vjp(model, x, dy, *, chunk=64):
    """Bounded full Y, dX and all atom dP, with task width stop-gradient."""
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    source = model.atoms.p.detach().cpu().double().numpy()
    xx, cotangent = (v.detach().cpu().double().numpy() for v in (x, dy))
    periods = model.chart.geometry.periods.detach().cpu().double().numpy()
    origins = model.chart.origin.detach().cpu().double().numpy()
    sizes = (model.out_features, model.in_features)
    profiles = tuple(binding.profile for binding in model.kernel.spec.profiles)
    sites = [
        o + np.arange(n, dtype=np.float64) * d / n
        for o, n, d in zip(origins, sizes, periods, strict=True)
    ]
    y, dx, gp = np.zeros_like(cotangent), np.zeros_like(xx), np.zeros_like(source)
    floor = model.kernel.spec.normalization.floor
    for start in range(0, len(source), chunk):
        p = source[start : start + chunk]
        amp, sigma, da = _polar(model.kernel, p)
        axes = [
            native_mixture(
                s[None, :] - p[:, dim + 2, None], sigma, d, profile.width_knots
            )
            for dim, (s, d, profile) in enumerate(
                zip(sites, periods, profiles, strict=True)
            )
        ]
        (u, du), (v, dv) = axes
        nu, nv = (u**2).sum(-1), (v**2).sum(-1)
        raw_norm = np.sqrt(nu * nv)
        norm = np.maximum(raw_norm, floor)
        scale = amp / norm
        h, g = xx @ v.T, cotangent @ u.T
        y += (h * scale[None, :]) @ u
        dx += (g * scale[None, :]) @ v
        dot = (h * g).sum(0)
        gp[start : start + len(p), :2] = (dot / norm)[:, None] * da
        norm_u = np.where(raw_norm >= floor, (u * du).sum(-1) / nu, 0)
        norm_v = np.where(raw_norm >= floor, (v * dv).sum(-1) / nv, 0)
        gp[start : start + len(p), 2] = scale * (
            (h * (cotangent @ du.T)).sum(0) - dot * norm_u
        )
        gp[start : start + len(p), 3] = scale * (
            ((xx @ dv.T) * g).sum(0) - dot * norm_v
        )
    return tuple(torch.from_numpy(v) for v in (y, dx, gp))
