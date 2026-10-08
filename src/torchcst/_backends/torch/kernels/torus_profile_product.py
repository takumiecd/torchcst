"""Exact centre-fibre chord product, including coupled centre derivatives."""

import torch

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles import execution as profiles
from torchcst.charts.base import _circle_axis

from .profile_product import _axis_positions


def factors(state, chart, p):
    _, centers = coordinates._split(state, chart, p)
    geometry = chart.geometry
    major, minor = geometry.major_radius.to(p), geometry.minor_radius.to(p)
    if geometry.representation == "intrinsic":
        theta = centers[:, 0] / major
        section = centers[:, 1:]
        angle = torch.linalg.vector_norm(section, dim=-1, keepdim=True) / minor
        q = torch.cat(
            (angle.cos(), torch.sinc(angle / torch.pi) * section / minor), dim=-1
        )
        radius = major + minor * q[:, 0]
    else:
        from torchcst._backends.torch.geometry import torus

        decoded = torus.decode_centers(geometry, centers)
        radius = torch.linalg.vector_norm(decoded[:, :2], dim=-1)
        theta = torch.atan2(decoded[:, 1], decoded[:, 0])
        q = torch.cat(
            (((radius - major) / minor)[:, None], decoded[:, 2:] / minor), dim=-1
        )
    circle_axis = _circle_axis(chart.spec)
    position = list(_axis_positions(chart))
    if circle_axis == 0:
        circle, section_axes = position[0], position[1:]
    else:
        circle, section_axes = position[-1], position[:-1]
    cross = torch.stack(torch.meshgrid(*section_axes, indexing="ij"), dim=-1)
    cross = cross.reshape(-1, 2)
    sites = torch.cat((minor.expand(len(cross), 1), cross), dim=-1)
    sites = sites / torch.linalg.vector_norm(sites, dim=-1, keepdim=True)
    # radius depends on q: retain this path in every section-centre VJP.
    dc = (
        4
        * radius[None, :].square()
        * (((circle[:, None] / major - theta[None, :]) / 2).sin().square())
    )
    ds = minor.square() * (sites[:, None, :] - q[None, :, :]).square().sum(-1)
    precision = coordinates.bandwidth_precision(state, chart, p).detach()
    raw = [
        profiles.shape_function(
            profile, "unnormalized_from_squared", distance, precision
        )
        for profile, distance in zip(state.profiles, (dc, ds), strict=True)
    ]
    amplitude = coordinates.amplitude(state, chart, p)
    dtype = torch.float32 if p.dtype in (torch.float16, torch.bfloat16) else p.dtype
    norm = torch.linalg.vector_norm(raw[0].to(dtype), dim=0)
    norm = norm * torch.linalg.vector_norm(raw[1].to(dtype), dim=0)
    scale = norm.clamp_min(state.spec.normalization.floor).sqrt()[None, :]
    output, source = raw if circle_axis == 0 else raw[::-1]
    return (source / scale).to(p.dtype), ((output / scale) * amplitude[None, :]).to(
        p.dtype
    )
