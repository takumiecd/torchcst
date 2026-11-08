"""Complete differentiable envelope map; no optimizer-history controller."""

import torch


def physical_parameters(p, *, radius_min, radius_max, tau, amplitude_max):
    if p.ndim != 2 or p.shape[1] != 4 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("radius envelope needs float32/float64 [atom,4] rows")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("radius envelope parameters must be finite")
    a, q = p[:, 0].double(), p[:, 1].double()
    low, high, scale, cap = (
        torch.as_tensor(v, dtype=torch.float64, device=p.device)
        for v in (radius_min, radius_max, tau, amplitude_max)
    )
    if bool(((q < 0) | (q > 1) | (a.abs() > cap)).any()):
        raise ValueError("radius envelope requires q in [0,1] and bounded amplitude")
    # Clip before division: exp(-32**2) is already zero in FP64. This avoids
    # overflow and 0*inf gradients for finite large amplitudes or tiny tau.
    z = a.clamp(-32 * scale, 32 * scale) / scale
    lower = low + (high - low) * (-z.square()).exp()
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
        **{
            name: state.scalar(name)
            for name in ("radius_min", "radius_max", "tau", "amplitude_max")
        },
    )
