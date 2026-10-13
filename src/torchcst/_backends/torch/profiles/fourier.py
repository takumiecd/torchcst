"""FP64 preparation of the native periodic profiles, with Torch only."""

import math

import torch

from torchcst.kernels.profiles.fourier import (
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
)


def effective_radius(spec, sigma, period):
    """Smooth concentration floor, reaching exactly one tile at sigma=tile."""
    if type(spec) is ChebyshevSquareSpec:
        a = math.acosh(spec.tail_bound**-0.5) / spec.modes
        minimum = period / math.pi * math.acos(1 / math.cosh(a))
    else:
        minimum = period * spec.concentration_floor / (spec.modes + 1)
    tile = period / 3
    ratio = minimum / tile
    return tile * (ratio**4 + (1 - ratio**4) * (sigma.double() / tile) ** 4) ** 0.25


def _chebyshev(spec, angle, sigma, period):
    radius = effective_radius(spec, sigma, period)
    beta = (math.pi * radius / period).cos().reciprocal()
    argument = beta * (angle / 2).cos()
    previous, current = torch.ones_like(argument), argument
    for _ in range(2, spec.modes + 1):
        previous, current = current, 2 * argument * current - previous
    return current / (spec.modes * beta.acosh()).cosh()


def amplitude_coefficients(spec, sigma, period):
    """Real symmetric H coefficients ordered -M/2..M/2; H(0)=1."""
    sigma = sigma.double()
    m = spec.modes // 2
    k = torch.arange(-m, m + 1, device=sigma.device, dtype=torch.float64)
    if type(spec) is ChebyshevSquareSpec:
        theta = (
            2
            * math.pi
            * torch.arange(spec.modes + 1, device=sigma.device, dtype=torch.float64)
            / (spec.modes + 1)
        )
        sample = _chebyshev(spec, theta[:, None], sigma[None, :], period)
        h = torch.fft.fftshift(torch.fft.fft(sample, dim=0), dim=0).real.T / (
            spec.modes + 1
        )
    else:
        nw = (spec.modes + 1) * effective_radius(spec, sigma, period) / period
        if type(spec) is KaiserSquareSpec:
            beta = math.pi * nw[:, None]
            z = beta * (1 - (k / m).square()).clamp_min(0).sqrt()[None, :]
            h = torch.special.i0e(z) / torch.special.i0e(beta) * (z - beta).exp()
        elif type(spec) is DPSSSquareSpec:
            # First DPSS eigenvector of the commuting symmetric tridiagonal.
            length = spec.modes + 1
            j = torch.arange(length, device=sigma.device, dtype=torch.float64)
            diagonal = ((length - 1 - 2 * j) / 2).square()[None, :] * (
                2 * math.pi * nw / length
            ).cos()[:, None]
            off = j[1:] * (length - j[1:]) / 2
            matrix = torch.diag_embed(diagonal)
            matrix = (
                matrix
                + torch.diag_embed(off, offset=1)
                + torch.diag_embed(off, offset=-1)
            )
            h = torch.linalg.eigh(matrix).eigenvectors[:, :, -1]
        else:
            raise ValueError("unsupported periodic Fourier profile")
    h = (h + h.flip(-1)) / 2
    return h / h.sum(-1, keepdim=True)


def kernel_coefficients(spec, sigma, period):
    """Linear convolution h*h, ordered -M..M (no circular aliasing)."""
    h = amplitude_coefficients(spec, sigma, period)
    length = 1 << (2 * spec.modes).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(h, n=length).square(), n=length)[
        ..., : 2 * spec.modes + 1
    ]
    return (c + c.flip(-1)) / 2


def values(spec, displacement, sigma, period):
    """Evaluate native H². No clipping and no approximation target."""
    angle = (2 * math.pi / period) * displacement.double()
    if type(spec) is ChebyshevSquareSpec:
        h_value = _chebyshev(spec, angle, sigma.double(), period)
    else:
        h = amplitude_coefficients(spec, sigma, period)
        k = torch.arange(
            -spec.modes // 2,
            spec.modes // 2 + 1,
            device=angle.device,
            dtype=torch.float64,
        )
        h_value = (torch.cos(angle[..., None] * k) * h).sum(-1)
    return h_value.square()
