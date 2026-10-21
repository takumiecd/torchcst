"""I-driven Fourier CPU reference, separate from CSTLinear dispatch.

This is a new explicit mathematical reference. Centered and whole-tile
support are different functions; selecting support never tunes a Recipe.
"""

import math

import torch

from torchcst._backends.torch.profiles.fourier_peak import coefficients
from torchcst.kernels.fourier_period import IntegerFourierPeriodSpec
from torchcst.kernels.normalization import NormalizationSpec


def _width(width):
    if not width.is_floating_point() or not bool(
        torch.isfinite(width).all() & (width > 0).all()
    ):
        raise ValueError("I must be a positive finite floating tensor")
    return width.double()


def period_components(period, width):
    """Return (...,2) L, weights, dweights/dI and minimum-safe integer k.

    For a fraction eta>0, blend over I in [k*L_min-T/2-eta*L_min,
    k*L_min-T/2]. Both periods are safe throughout; the next period has
    weight one at the minimal-k switch itself.
    """
    if type(period) is not IntegerFourierPeriodSpec:
        raise TypeError("period must be IntegerFourierPeriodSpec")
    width = _width(width)
    step, tile = period.minimum_half_period, period.tile_width
    scaled = (width + tile / 2) / step
    k = scaled.ceil().clamp_min(1)
    low = k * step
    eta = period.transition_fraction
    if eta:
        z = ((width - (low - tile / 2 - eta * step)) / (eta * step)).clamp(0, 1)
        w = z**3 * (10 + z * (-15 + 6 * z))
        dw = 30 * z.square() * (1 - z).square() / (eta * step)
    else:
        w, dw = torch.zeros_like(width), torch.zeros_like(width)
    return (
        torch.stack((low, low + step), -1),
        torch.stack((1 - w, w), -1),
        torch.stack((-dw, dw), -1),
        k,
    )


def tile_bounds(period, center, width, *, origin=0.0):
    """Continuous effective region covered by complete physical tiles."""
    width = _width(width)
    tile = period.tile_width
    return (
        origin + tile * torch.floor((center - width - origin) / tile),
        origin + tile * torch.ceil((center + width - origin) / tile),
    )


def values_and_derivatives(
    profile, period, sites, center, width, *, support, normalization=None, origin=0.0
):
    """Return [sites,...] value, d/dcenter, d/dI and raw sampled norm.

    Autograd through returned values includes coefficient and transition
    derivatives. Analytic derivatives follow the currently selected integer
    routing. At a whole-tile support jump, a classical derivative need not
    exist; the branch VJP is not a claim of global continuity.
    """
    width = _width(width)
    center, width = torch.broadcast_tensors(center.double(), width)
    if (
        sites.ndim != 1
        or not sites.is_floating_point()
        or not bool(torch.isfinite(sites).all())
    ):
        raise ValueError("sites must be a finite floating vector")
    if not bool(torch.isfinite(center).all()):
        raise ValueError("center must be finite")
    if support not in ("centered", "whole_tiles"):
        raise ValueError("support must be centered or whole_tiles")
    norm_spec = normalization or NormalizationSpec()
    if not isinstance(norm_spec, NormalizationSpec) or (
        norm_spec.kind != "none"
        and (
            norm_spec.kind != "discrete_l2"
            or norm_spec.domain != "chart_sites"
            or norm_spec.floor is None
        )
    ):
        raise ValueError(
            "reference normalization requires none or chart_sites discrete_l2 with a floor"
        )
    L, weights, derivatives, _ = period_components(period, width)
    b, bq = coefficients(profile, width[..., None] / L)
    shift = sites.double().reshape((-1,) + (1,) * center.ndim) - center
    k = torch.arange(profile.modes + 1, dtype=torch.float64, device=sites.device)
    omega = math.pi * k / L[..., None]
    theta = shift[..., None, None] * omega
    cosine = theta.cos()
    raw = (cosine * b).sum(-1)
    dm = (theta.sin() * omega * b).sum(-1)
    di = (cosine * (bq / L[..., None])).sum(-1)
    if support == "centered":
        mask = shift.abs()[..., None] < L
    else:
        left, right = tile_bounds(period, center, width, origin=origin)
        x = sites.double().reshape((-1,) + (1,) * center.ndim)
        mask = ((x >= left) & (x < right))[..., None]
    value = torch.where(mask, raw * weights, 0.0).sum(-1)
    center_grad = torch.where(mask, dm * weights, 0.0).sum(-1)
    width_grad = torch.where(mask, di * weights + raw * derivatives, 0.0).sum(-1)
    norm = torch.linalg.vector_norm(value, dim=0)
    if norm_spec.kind == "discrete_l2":
        denominator = norm.clamp_min(norm_spec.floor)
        live = norm >= norm_spec.floor
        center_grad = (
            center_grad / denominator
            - value * (value * center_grad).sum(0) * live / denominator**3
        )
        width_grad = (
            width_grad / denominator
            - value * (value * width_grad).sum(0) * live / denominator**3
        )
        value = value / denominator
    return value, center_grad, width_grad, norm


def values(
    profile, period, sites, center, width, *, support, normalization=None, origin=0.0
):
    return values_and_derivatives(
        profile,
        period,
        sites,
        center,
        width,
        support=support,
        normalization=normalization,
        origin=origin,
    )[0]
