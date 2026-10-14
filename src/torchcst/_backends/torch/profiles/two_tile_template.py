"""Native positive Kaiser templates; no approximation to another profile."""

import math

import torch

from torchcst.kernels.profiles.fourier import KaiserTemplateSpec


def effective_radius(spec, sigma, period):
    """Smooth width floor, reaching R=P/4 at sigma=R."""
    minimum = spec.concentration_floor * period / (spec.modes + 1)
    radius = period / 4
    return (minimum**4 + (1 - (minimum / radius) ** 4) * sigma.double() ** 4) ** 0.25


def static_tables(spec, period, *, device=None):
    """Return h, linear-convolution c, and adjacent coefficient Gram entries.

    Tables are immutable declaration data in FP64. Gram columns are
    (||c_j||², <c_j,c_{j+1}>, ||c_{j+1}||²); multiply by the complete-period
    site count for the exact sampled squared norm when N>2M.
    """
    if type(spec) is not KaiserTemplateSpec or spec.revision != 1:
        raise ValueError("unsupported two-tile template profile")
    if not math.isfinite(period) or period <= 0 or spec.width_knots[-1] > period / 4:
        raise ValueError(
            "template period must be positive and contain the declared widths"
        )
    sigma = torch.tensor(spec.width_knots, dtype=torch.float64, device=device)
    k = torch.arange(
        -spec.modes // 2, spec.modes // 2 + 1, dtype=torch.float64, device=device
    )
    beta = math.pi * (spec.modes + 1) * effective_radius(spec, sigma, period) / period
    z = beta[:, None] * (1 - (2 * k / spec.modes).square()).clamp_min(0).sqrt()
    h = (
        torch.special.i0e(z)
        / torch.special.i0e(beta[:, None])
        * (z - beta[:, None]).exp()
    )
    h = (h + h.flip(-1)) / 2
    h = h / h.sum(-1, keepdim=True)
    # Zero padding gives linear convolution once per declaration. There is no
    # atom-wise FFT: each live width only selects and blends two table rows.
    length = 1 << (2 * spec.modes).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(h, n=length).square(), n=length)[
        ..., : 2 * spec.modes + 1
    ]
    c = (c + c.flip(-1)) / 2
    gram = torch.stack(
        (c[:-1].square().sum(-1), (c[:-1] * c[1:]).sum(-1), c[1:].square().sum(-1)),
        dim=-1,
    )
    return h, c, gram


def interpolation(spec, sigma):
    """Clamped log-width interval and convex fraction, with device-local lookup."""
    sigma = sigma.double()
    knots = sigma.new_tensor(spec.width_knots)
    bounded = sigma.clamp(knots[0], knots[-1])
    index = (torch.searchsorted(knots, bounded.contiguous(), right=True) - 1).clamp(
        0, len(spec.width_knots) - 2
    )
    lower, upper = knots[index], knots[index + 1]
    fraction = (bounded / lower).log() / (upper / lower).log()
    return index, fraction


def kernel_coefficients(spec, sigma, period):
    """Blend c_j and c_{j+1}; ordered -M..M."""
    _, table, _ = static_tables(spec, period, device=sigma.device)
    index, fraction = interpolation(spec, sigma)
    return (
        table[index] * (1 - fraction[..., None])
        + table[index + 1] * fraction[..., None]
    )


def values(spec, displacement, sigma, period):
    """Evaluate (1-t) H_j² + t H_{j+1}² directly, preserving positivity."""
    table, _, _ = static_tables(spec, period, device=sigma.device)
    index, fraction = interpolation(spec, sigma)
    k = torch.arange(
        -spec.modes // 2,
        spec.modes // 2 + 1,
        dtype=torch.float64,
        device=displacement.device,
    )
    basis = torch.cos(displacement.double()[..., None] * (2 * math.pi / period) * k)
    lower = (basis * table[index]).sum(-1).square()
    upper = (basis * table[index + 1]).sum(-1).square()
    return lower * (1 - fraction) + upper * fraction
