"""Decode rational radius fractions with the historical endpoint Jacobian."""

import torch


def physical_parameters(p, radius_min, radius_birth, radius_max, amplitude_max):
    if p.ndim != 2 or p.shape[1] != 4 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("radius envelope needs float32/float64 [atom,4] rows")
    a, q = p[:, 0].double(), p[:, 1].double()
    low, birth, high, cap = (
        torch.as_tensor(v, dtype=torch.float64, device=p.device)
        for v in (radius_min, radius_birth, radius_max, amplitude_max)
    )
    invalid = (~torch.isfinite(p)).any() | ((q < 0) | (q > 1) | (a.abs() > cap)).any()
    if bool(invalid):
        raise ValueError(
            "rational radius requires finite rows, q in [0,1] and bounded amplitude"
        )
    lower = low + (birth - low) / (1.0 + 3.0 * (a / 0.3).square())
    raw = ((1 - q) * lower.log() + q * high.log()).exp()
    # Exact endpoint values, with the one-sided/interior analytic Jacobian.
    value = torch.where(q == 0, lower, torch.where(q == 1, high, raw))
    raw = raw + (value - raw).detach()
    # Encoded physical rows must remain inside the exact declared interval,
    # including a nonrepresentable FP32 chart half-period.
    lo, hi = low.to(p), high.to(p)
    lo = torch.where(
        lo.double() < low, torch.nextafter(lo, torch.full_like(lo, float("inf"))), lo
    )
    hi = torch.where(hi.double() > high, torch.nextafter(hi, torch.zeros_like(hi)), hi)
    if bool(lo > hi):
        raise ValueError(
            "radius interval has no representable value in parameter dtype"
        )
    encoded = raw.to(p)
    bounded = encoded.clamp(lo, hi)
    radius = encoded + (bounded - encoded).detach()
    return torch.stack((p[:, 0], radius, p[:, 2], p[:, 3]), dim=1)


def from_state(state, p):
    return physical_parameters(
        p,
        state.scalar("radius_min"),
        state.scalar("radius_birth"),
        state.scalar("radius_max"),
        state.scalar("amplitude_max"),
    )
