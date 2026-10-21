"""Torch reference for the sigma-driven finite Fourier function (FP64 prep).

Only fixed projection matrices are cached. Widths, routes, coefficients and
norms are rebuilt from live tensors. No benchmark imports or GPU executor.
"""

import math
from functools import cache

import torch

from torchcst.kernels.profiles.dyadic_fourier import DyadicFourierSpec

# Reuse geometry, normalization and state-update contracts. These functions
# dispatch shape evaluation through profiles.execution back into this module.
from .compact import (  # noqa: F401
    apply_parameter_update,
    evaluate,
    evaluate_with_precision,
    evaluate_with_precision_slice,
    initialize,
    project_gradient,
    tangent,
    tangent_with_precision,
    transport_state,
)


def _validate(spec, sigma):
    if type(spec) is not DyadicFourierSpec or spec.revision != 1:
        raise ValueError("unsupported dyadic Fourier profile or revision")
    if not sigma.is_floating_point() or not bool(
        torch.isfinite(sigma).all() & (sigma > 0).all()
    ):
        raise ValueError("sigma must be a positive finite floating tensor")


def scale_weights(spec, sigma):
    """Return (...,2) half-widths, weights and analytic dweights/dsigma.

    Inactive second components have zero weight and derivative. Integer
    routing has zero derivative; the quintic transition has two flat ends.
    """
    _validate(spec, sigma)
    sigma = sigma.double()
    radius = math.sqrt(11) * sigma
    relative = torch.log2(radius / (spec.level_ratio * spec.minimum_half_width))
    # Check the lower dyadic boundary first. Logarithmic rounding alone can
    # split a wide (but nonoverlapping) transition before its flat endpoint.
    below = relative.floor().clamp_min(0)
    below_boundary = spec.level_ratio * spec.minimum_half_width * torch.exp2(below)
    in_below = (radius >= below_boundary * (1 - spec.overlap)) & (
        radius <= below_boundary * (1 + spec.overlap)
    )
    nearest = torch.where(in_below, below, relative.ceil().clamp_min(0))
    boundary = spec.level_ratio * spec.minimum_half_width * torch.exp2(nearest)
    lower, upper = boundary * (1 - spec.overlap), boundary * (1 + spec.overlap)
    mixing = (radius >= lower) & (radius <= upper)
    level = torch.where(mixing, nearest, relative.ceil().clamp_min(0))
    half_width = spec.minimum_half_width * torch.exp2(level)
    z = ((radius - lower) / (upper - lower)).clamp(0, 1)
    weight = torch.where(mixing, z**3 * (10 + z * (-15 + 6 * z)), 0.0)
    derivative = torch.where(
        mixing,
        30 * z.square() * (1 - z).square() * math.sqrt(11) / (upper - lower),
        0.0,
    )
    return (
        torch.stack((half_width, 2 * half_width), -1),
        torch.stack((1 - weight, weight), -1),
        torch.stack((-derivative, derivative), -1),
    )


@cache
@torch.inference_mode(False)
def _projection(modes, device):
    k = torch.arange(modes + 1, dtype=torch.float64, device=device)
    zeros = 2 if modes == 4 else 3
    a = torch.stack(
        [(-1.0) ** k * (k / modes) ** (2 * j) for j in range(zeros)]
        + [torch.ones_like(k)]
    )
    inv_metric = torch.diag(torch.where(k == 0, 0.5, 1.0).double())
    right = torch.linalg.solve(a @ inv_metric @ a.T, a @ inv_metric).T
    return torch.eye(modes + 1, dtype=k.dtype, device=device) - right @ a, right[:, -1]


def cosine_coefficients(spec, sigma, half_width):
    """Unweighted (...,M+1) cosine coefficients and their d/dsigma at fixed L.

    Uses the analytic transform of (1-u*u)^4 on [-1,1]. Endpoint conditions
    and peak-one are a fixed continuous-L2 projection in coefficient space.
    """
    _validate(spec, sigma)
    sigma, half_width = torch.broadcast_tensors(sigma.double(), half_width.double())
    if not bool(torch.isfinite(half_width).all() & (half_width > 0).all()):
        raise ValueError("half_width must be positive and finite")
    k = torch.arange(spec.modes + 1, dtype=sigma.dtype, device=sigma.device)
    radius = math.sqrt(11) * sigma
    z = math.pi * radius[..., None] / half_width[..., None] * k
    small = z.abs() < 4
    t = torch.where(small, z, 0.0)
    term = torch.full_like(t, 256 / 315)
    integral, zprime = term, torch.zeros_like(term)
    for n in range(20):
        term = term * (-t.square() / ((2 * n + 2) * (2 * n + 11)))
        integral = integral + term
        zprime = zprime + (2 * n + 2) * term
    # Safe unselected branch: no division by zero in autograd.
    v = torch.where(small, 4.0, z)
    v2 = v.square()
    large = (
        768
        * ((105 - 45 * v2 + v2.square()) * v.sin() - (105 * v - 10 * v * v2) * v.cos())
        / v**9
    )
    large_prime = (
        -768
        * (
            (945 - 420 * v2 + 15 * v2.square()) * v.sin()
            - (945 * v - 105 * v * v2 + v * v2.square()) * v.cos()
        )
        / v**9
    )
    integral = torch.where(small, integral, large)
    zprime = torch.where(small, zprime, large_prime)
    factor = torch.where(k == 0, 1.0, 2.0) / (2 * half_width[..., None])
    c = factor * radius[..., None] * integral
    cs = factor * math.sqrt(11) * (integral + zprime)
    matrix, offset = _projection(spec.modes, sigma.device)
    return c @ matrix.T + offset, cs @ matrix.T


def components(spec, sigma):
    """Return L, weighted cosine coefficients and total sigma derivatives.

    Shapes are (...,2), (...,2,M+1), (...,2,M+1). No center phase is applied;
    an executor can reuse the same moment basis for all centers at a scale.
    """
    half_width, weights, derivatives = scale_weights(spec, sigma)
    c, cs = cosine_coefficients(spec, sigma[..., None], half_width)
    return (
        half_width,
        weights[..., None] * c,
        derivatives[..., None] * c + weights[..., None] * cs,
    )


def _from_squared(spec, squared, sigma):
    half_width, c, cs = components(spec, sigma)
    k = torch.arange(spec.modes + 1, dtype=torch.float64, device=squared.device)
    omega2 = (math.pi * k / half_width[..., None]).square()
    u = squared.double()[..., None, None] * omega2
    small = u < 1e-4
    v = torch.sqrt(torch.where(small, 1.0, u))
    # An even power series preserves both derivatives at exactly zero distance.
    s = torch.where(small, u, 0.0)
    basis = torch.where(
        small, 1 - s / 2 + s**2 / 24 - s**3 / 720 + s**4 / 40320, v.cos()
    )
    dq = (
        torch.where(
            small, -0.5 + s / 12 - s**2 / 240 + s**3 / 10080, -0.5 * v.sin() / v
        )
        * omega2
    )
    active = squared.double()[..., None] < half_width.square()
    return tuple(
        (torch.where(active, expression.sum(-1), 0.0)).sum(-1)
        for expression in (basis * c, dq * c, basis * cs)
    )


def values(spec, displacement, sigma):
    """Raw FP64 values, broadcasting signed displacement against sigma.

    L2 remains an explicit separate normalization operation on actual sites.
    This reference uses Tensor autograd, including the sigma transition.
    """
    return _from_squared(spec, displacement.double().square(), sigma)[0]


def dyadic_fourier_unnormalized_from_squared(state, squared, precision):
    return _from_squared(state.binding.profile, squared, precision.rsqrt())[0].to(
        squared.dtype
    )


def dyadic_fourier_d_raw_d_center(state, offset, squared, precision):
    dq = _from_squared(state.binding.profile, squared, precision.rsqrt())[1]
    return (-2 * offset * dq[..., None]).to(offset.dtype)


def dyadic_fourier_d_raw_d_precision(state, squared, precision):
    ds = _from_squared(state.binding.profile, squared, precision.rsqrt())[2]
    return (ds * (-0.5 * precision.double().pow(-1.5))).to(squared.dtype)
