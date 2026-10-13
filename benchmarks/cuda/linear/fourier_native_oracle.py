"""Bounded full-site CPU FP64 truth for equal-tile Fourier density cases.

The native H function has degree M/2. Its coefficients are independently
declared with NumPy, evaluated at every site, and squared before computing the
actual discrete norms. This oracle never consumes CUDA preparation, K=H*H
coefficients, shared tile moments, or a centre-independent norm identity.
"""

import copy
import math

import numpy as np
import torch

from benchmarks.cuda.linear.periodic_profile_product import polar_values
from torchcst import ChebyshevSquareSpec, KaiserSquareSpec


def native_h_batch(profile, sigma, period):
    """Vectorized independent native H declarations, one row per width."""
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    m, tile = profile.modes, period / 3
    if isinstance(profile, ChebyshevSquareSpec):
        r0 = (
            period
            / math.pi
            * math.acos(1 / math.cosh(math.acosh(profile.tail_bound**-0.5) / m))
        )
    elif isinstance(profile, KaiserSquareSpec):
        r0 = period * profile.concentration_floor / (m + 1)
    else:
        raise TypeError("density native oracle supports Chebyshev and Kaiser")
    radius = (r0**4 + (1 - (r0 / tile) ** 4) * sigma**4) ** 0.25
    if isinstance(profile, ChebyshevSquareSpec):
        beta = 1 / np.cos(math.pi * radius / period)
        q = beta[:, None] * np.cos(np.arange(m + 1) * math.pi / (m + 1))
        native = np.where(
            abs(q) <= 1,
            np.cos(m * np.arccos(np.clip(q, -1, 1))),
            np.cosh(m * np.arccosh(np.maximum(abs(q), 1))),
        )
        native /= np.cosh(m * np.arccosh(beta))[:, None]
        h = np.fft.fftshift(np.fft.fft(native, axis=-1), axes=-1).real / (m + 1)
    else:
        beta = math.pi * (m + 1) * radius / period
        k = np.arange(-m // 2, m // 2 + 1)
        h = (
            np.i0(beta[:, None] * np.sqrt(np.maximum(1 - (2 * k / m) ** 2, 0)))
            / np.i0(beta)[:, None]
        )
    h = (h + h[:, ::-1]) / 2
    return h / h.sum(axis=-1, keepdims=True)


def _site_bases(size, period, tile, modes):
    """Native H site bases for every owner, including masked-out sites."""
    tiles = round(period / tile)
    if tiles < 3 or size % tiles or not math.isclose(period, tiles * tile):
        raise ValueError("oracle requires a full grid of at least three equal tiles")
    sites = size // tiles
    site_tile = torch.arange(size) // sites
    offsets = (torch.arange(size) % sites).double() * (period / size)
    relative = (site_tile[None, :] - torch.arange(tiles)[:, None]).remainder(tiles)
    relative = torch.where(relative == tiles - 1, -1, relative)
    mask = ((relative >= -1) & (relative <= 1)).double()
    distance = offsets[None, :] + tile * relative.double()
    omega = torch.arange(1, modes // 2 + 1).double() * (2 * math.pi / (3 * tile))
    phase = distance[:, None, :] * omega[None, :, None]
    return phase.cos(), phase.sin(), mask, omega


def _native_axis(p, sigma, profile, dim, period, tile, origin, bases):
    cos_site, sin_site, mask, omega = bases
    center = p[:, dim + 2] - origin
    wrapped = center - torch.floor(center / period).detach() * period
    owner = torch.floor(wrapped / tile).detach().long()
    delta = wrapped - owner.double() * tile
    h = p.new_tensor(native_h_batch(profile, sigma.numpy(), 3 * tile))
    degree = profile.modes // 2
    phase = delta[:, None] * omega[None, :]
    cos_coeff = 2 * h[:, degree + 1 :] * phase.cos()
    sin_coeff = 2 * h[:, degree + 1 :] * phase.sin()
    # Grouped matmul avoids materializing an atom x site x harmonic tensor.
    raw = p.new_zeros((len(p), mask.shape[-1]))
    for index in range(len(mask)):
        selected = torch.nonzero(owner == index, as_tuple=True)[0]
        if not len(selected):
            continue
        hh = (
            cos_coeff[selected] @ cos_site[index]
            + sin_coeff[selected] @ sin_site[index]
            + h[selected, degree, None]
        )
        raw = raw.index_copy(0, selected, hh.square() * mask[index])
    return raw


def oracle_vjp(model, x, dy, *, chunk=64):
    """Return full Y, dX, and all atom gradients on CPU in FP64.

    Width is task-stop-gradient, while amplitude and centre derivatives retain
    the full native site evaluation and norm/floor graph. Peak scratch depends
    on the chunk size, native H degree, batch, and sites, not total atom count.
    Caller controls CPU thread policy, so this helper does not mutate it.
    """
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    kernel = copy.deepcopy(model.kernel).to("cpu", torch.float64)
    tx = x.detach().cpu().double().requires_grad_()
    cotangent = dy.detach().cpu().double()
    periods = model.chart.geometry.periods.detach().cpu().double().tolist()
    origins = model.chart.origin.detach().cpu().double().tolist()
    tile_widths = kernel.spec.tile_support.tile_widths
    sizes = (model.out_features, model.in_features)
    profiles = [binding.profile for binding in kernel.spec.profiles]
    bases = [
        _site_bases(size, period, tile, profile.modes)
        for size, period, tile, profile in zip(sizes, periods, tile_widths, profiles)
    ]
    source = model.atoms.p.detach().cpu().double()
    y, dx, gradients = torch.zeros_like(cotangent), torch.zeros_like(tx), []
    for start in range(0, model.atom_count, chunk):
        p = source[start : start + chunk].clone().requires_grad_()
        amp, sigma = polar_values(kernel, p)
        axes = [
            _native_axis(p, sigma, profile, dim, period, tile, origin, axis_bases)
            for dim, (profile, period, tile, origin, axis_bases) in enumerate(
                zip(profiles, periods, tile_widths, origins, bases)
            )
        ]
        u, v = axes
        norm = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(
            kernel.spec.normalization.floor
        )
        yy = ((tx @ v.T) * (amp / norm)[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, p), cotangent)
        y += yy.detach()
        dx += gx
        gradients.append(gp)
    gp = torch.cat(gradients) if gradients else torch.zeros_like(source)
    return y, dx, gp
