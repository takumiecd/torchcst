"""Native actual-site reference; one norm floor after the Cartesian product."""

import torch

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.matern_mixture import values
from torchcst.charts import RegularGridChartState


def factors(state, chart, p):
    from .profile_product import _axis_positions, _product

    _, centres = coordinates._split(state, chart, p)
    sigma = coordinates.bandwidth_sigma(state, chart, p).detach()
    amp = coordinates.amplitude(state, chart, p)
    axes = [
        values(
            profile.binding.profile,
            positions.double()[:, None] - centres.double()[None, :, dim],
            sigma,
            chart.geometry.periods[dim].double(),
        ).to(p.dtype)
        for dim, (positions, profile) in enumerate(
            zip(_axis_positions(chart), state.profiles)
        )
    ]
    dtype = torch.float32 if p.dtype in (torch.float16, torch.bfloat16) else p.dtype
    norm = torch.ones_like(amp, dtype=dtype)
    for axis in axes:
        norm = norm * torch.linalg.vector_norm(axis.to(dtype), dim=0)
    scale = norm.clamp_min(state.spec.normalization.floor).sqrt()[None, :]
    out_dims = (
        len(chart.grid_shape[0])
        if type(chart) is RegularGridChartState
        else chart.output_dims
    )
    return (
        (_product(axes[out_dims:]) / scale).to(p.dtype),
        ((_product(axes[:out_dims]) / scale) * amp[None, :]).to(p.dtype),
    )
