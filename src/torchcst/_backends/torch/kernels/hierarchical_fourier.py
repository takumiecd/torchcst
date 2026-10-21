"""CPU mathematical reference for integer-k hierarchies, not CST dispatch."""

import torch

from torchcst.kernels.fourier_period import HierarchicalFourierPeriodSpec

from .effective_fourier import _width, evaluate_components


def _smoothstep(z):
    z = z.clamp(0, 1)
    return z**3 * (10 + z * (-15 + 6 * z)), 30 * z.square() * (1 - z).square()


def level_metadata(spec, *, device=None):
    if type(spec) is not HierarchicalFourierPeriodSpec:
        raise TypeError("spec must be HierarchicalFourierPeriodSpec")
    scale = 2.0 ** torch.arange(spec.level_count, dtype=torch.float64, device=device)
    tile = spec.minimum_tile_width * scale
    stride = scale if spec.k_stride == "dyadic" else torch.ones_like(scale)
    step = spec.minimum_half_period * stride
    threshold = (tile / 2 + (1 + spec.period_transition_fraction) * step) / (
        1 / spec.relative_width_floor - 1
    )
    return tile, stride, step, threshold


def components(spec, width):
    """Return L,T,weight,dweight/dI,k,level with final axis of length four.

    The last axis is two eligible hierarchy levels times two period
    components. Zero-weight entries have no semantic support.
    The base level must already be eligible for every supplied I.
    """
    width = _width(width)
    tile, stride, step, threshold = level_metadata(spec, device=width.device)
    if bool((width < threshold[0]).any()):
        raise ValueError(
            "I is below the eligible base level; choose smaller T_min or L_min"
        )
    level = ((width[..., None] >= threshold).sum(-1) - 1).clamp_max(
        spec.level_count - 1
    )
    coarse, fine = level, (level - 1).clamp_min(0)
    gamma = spec.hierarchy_transition_fraction
    if gamma:
        interval = gamma * (threshold[coarse] - threshold[fine])
        interval = torch.where(coarse > 0, interval, torch.ones_like(interval))
        h, dh = _smoothstep((width - threshold[coarse]) / interval)
        dh = dh / interval
        h = torch.where(coarse > 0, h, 1.0)
        dh = torch.where(coarse > 0, dh, 0.0)
    else:
        h, dh = torch.ones_like(width), torch.zeros_like(width)
    levels = torch.stack((fine, coarse), -1)
    outer = torch.stack((1 - h, h), -1)
    douter = torch.stack((-dh, dh), -1)
    t, s, r = tile[levels], step[levels], stride[levels]
    n = ((width[..., None] + t / 2) / s).ceil().clamp_min(1)
    low = n * s
    eta = spec.period_transition_fraction
    if eta:
        w, dw = _smoothstep((width[..., None] - (low - t / 2 - eta * s)) / (eta * s))
        dw = dw / (eta * s)
    else:
        w, dw = torch.zeros_like(low), torch.zeros_like(low)
    inner = torch.stack((1 - w, w), -1)
    dinner = torch.stack((-dw, dw), -1)
    lengths = torch.stack((low, low + s), -1)
    global_k = torch.stack((n * r, (n + 1) * r), -1)
    weight = outer[..., None] * inner
    derivative = douter[..., None] * inner + outer[..., None] * dinner
    shape = width.shape + (4,)
    return (
        lengths.reshape(shape),
        t[..., None].expand_as(lengths).reshape(shape),
        weight.reshape(shape),
        derivative.reshape(shape),
        global_k.reshape(shape),
        levels[..., None].expand_as(lengths).reshape(shape),
    )


def values_and_derivatives(
    profile, spec, sites, center, width, *, support, normalization=None, origin=0.0
):
    L, tiles, weights, derivatives, _, _ = components(spec, width)
    return evaluate_components(
        profile,
        sites,
        center,
        width,
        L,
        tiles,
        weights,
        derivatives,
        support=support,
        normalization=normalization,
        origin=origin,
    )


def values(
    profile, spec, sites, center, width, *, support, normalization=None, origin=0.0
):
    return values_and_derivatives(
        profile,
        spec,
        sites,
        center,
        width,
        support=support,
        normalization=normalization,
        origin=origin,
    )[0]
