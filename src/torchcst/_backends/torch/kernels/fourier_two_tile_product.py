"""Direct native-site reference for the revision-5 two-tile product."""

import torch

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.two_tile_template import values
from torchcst.charts import RegularGridChartState


def axis_geometry(chart, centers, dim, width, size):
    """Two-tile virtual coordinates; midpoint tie chooses left+owner."""
    period = chart.geometry.periods[dim].double()
    center = (centers[:, dim].double() - chart.origin[dim].double()).remainder(period)
    owner = torch.floor(center / width).to(torch.long)
    delta = center - owner.to(center.dtype) * width
    left = delta <= width / 2
    tiles = (period / width).round().to(torch.long)
    pair = (owner - left.to(torch.long)).remainder(tiles)
    pair_center = delta + left.to(delta.dtype) * width
    indices = torch.arange(size, device=center.device, dtype=torch.float64)
    spacing = (
        chart.spacing[dim].double()
        if type(chart) is RegularGridChartState
        else period / size
    )
    position = indices * spacing
    site_tile = torch.floor(position / width).to(torch.long)
    relative = (site_tile[:, None] - pair[None, :]).remainder(tiles)
    displacement = (
        position[:, None]
        - site_tile[:, None].to(position.dtype) * width
        + relative.to(position.dtype) * width
        - pair_center[None, :]
    )
    return displacement, relative < 2


def factors(state, chart, p):
    from .profile_product import _product

    _, centers = coordinates._split(state, chart, p)
    sigma = coordinates.bandwidth_sigma(state, chart, p).detach()
    amp = coordinates.amplitude(state, chart, p)
    regular = type(chart) is RegularGridChartState
    sizes = chart.coordinate_shape if regular else chart.grid_shape
    output_dims = len(chart.grid_shape[0]) if regular else chart.output_dims
    axes = []
    for dim, (size, width, profile) in enumerate(
        zip(sizes, state.spec.tile_support.tile_widths, state.profiles)
    ):
        delta, mask = axis_geometry(chart, centers, dim, width, size)
        raw = values(profile.binding.profile, delta, sigma, 2 * width)
        axes.append(torch.where(mask, raw, 0).to(p.dtype))
    norm_dtype = (
        torch.float32 if p.dtype in (torch.float16, torch.bfloat16) else p.dtype
    )
    norm = torch.ones_like(amp, dtype=norm_dtype)
    for axis in axes:
        norm = norm * torch.linalg.vector_norm(axis.to(norm_dtype), dim=0)
    scale = norm.clamp_min(state.spec.normalization.floor).sqrt()[None, :]
    return (
        (_product(axes[output_dims:]) / scale).to(p.dtype),
        ((_product(axes[:output_dims]) / scale) * amp[None, :]).to(p.dtype),
    )
