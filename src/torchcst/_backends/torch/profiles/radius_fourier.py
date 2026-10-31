"""CPU FP64 reference for radius-input Chebyshev cell probabilities.

This module intentionally builds temporary coefficients for an independent
reference. It specifies no global coefficient table for the future CUDA path.
"""

import math

import torch

from torchcst.kernels.profiles.fourier import ChebyshevRadiusSpec


def density_coefficients(profile, radius, period):
    """Return normalized Fourier coefficients ordered -M..M.

    The scaled Chebyshev recurrence avoids constructing a large secant or
    dividing two overflowing Chebyshev polynomials. Squaring the sampled real
    polynomial gives a density of degree M even when M is odd.
    """
    if type(profile) is not ChebyshevRadiusSpec:
        raise TypeError("expected ChebyshevRadiusSpec")
    if radius.device.type != "cpu":
        raise ValueError("radius cell-mass reference currently requires CPU tensors")
    if radius.ndim != 1 or radius.dtype not in (torch.float32, torch.float64):
        raise ValueError("radius must be a floating-point vector")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("profile period must be finite and positive")
    if not bool(torch.isfinite(radius).all()) or bool(
        ((radius <= 0) | (radius > period / 2)).any()
    ):
        raise ValueError("radius must lie in (0, profile period/2]")
    modes = profile.modes
    if radius.numel() == 0:
        return radius.double()[:, None].expand(-1, 2 * modes + 1)
    theta = (
        2
        * torch.pi
        * torch.arange(2 * modes + 1, dtype=torch.float64)
        / (2 * modes + 1)
    )
    cosine = (theta / 2).cos()[None, :]
    inverse_beta_squared = (torch.pi * radius.double()[:, None] / period).cos().square()
    previous = torch.full_like(inverse_beta_squared + cosine, 2)
    current = cosine.expand(radius.numel(), -1)
    for _ in range(2, modes + 1):
        previous, current = (
            current,
            cosine * current - inverse_beta_squared * previous / 4,
        )
    scale = current.abs().amax(dim=-1, keepdim=True).detach()
    density = (current / scale).square()
    coefficients = torch.fft.fftshift(torch.fft.fft(density, dim=-1), dim=-1).real / (
        2 * modes + 1
    )
    # Enforce the exact even symmetry, as in the existing sigma reference.
    coefficients = (coefficients + coefficients.flip(-1)) / 2
    return coefficients / (period * coefficients[:, modes, None])


def integrated_cells(profile, radius, centers, period, sites, spacing):
    """Integrate whole unwrapped cells, without per-cell support clipping."""
    coefficients = density_coefficients(profile, radius, period)
    modes = profile.modes
    harmonic = torch.arange(1, modes + 1, dtype=torch.float64)
    displacement = sites.double() * spacing - centers.double()[:, None]
    phase = (2 * torch.pi / period) * displacement[..., None] * harmonic
    cell_filter = torch.sinc(harmonic * spacing / period)
    return spacing * (
        coefficients[:, modes, None]
        + 2 * (coefficients[:, None, modes + 1 :] * cell_filter * phase.cos()).sum(-1)
    )


def support_eps(profile, radius, period):
    """Mass outside [-radius,+radius]; diagnostic, not an input tolerance."""
    coefficients = density_coefficients(profile, radius, period)
    modes = profile.modes
    omega = (2 * torch.pi / period) * torch.arange(1, modes + 1, dtype=torch.float64)
    enclosed = 2 * radius.double() * coefficients[:, modes] + (
        4
        * coefficients[:, modes + 1 :]
        * (radius.double()[:, None] * omega).sin()
        / omega
    ).sum(-1)
    # Only diagnostic probability bounds absorb final FP64 roundoff.
    return (1 - enclosed).clamp(0, 1)
