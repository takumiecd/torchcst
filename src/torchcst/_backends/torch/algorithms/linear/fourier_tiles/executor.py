"""Complex moments make the k±l/phase identities explicit in the reference."""

import math

import torch

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.fourier import kernel_coefficients


def _axis(chart, centers, sigma, dim, layout, profile):
    _tiles, sites, width = layout
    period = 3 * width
    center = (centers[:, dim] - chart.origin[dim]).remainder(
        chart.geometry.periods[dim]
    )
    owner = (center / width).floor().to(torch.long)
    delta = center - owner * width
    k = torch.arange(
        -profile.modes, profile.modes + 1, device=center.device, dtype=center.dtype
    )
    omega = 2 * math.pi / period
    c = kernel_coefficients(profile, sigma, period)
    beta = c * torch.exp(-1j * omega * delta[:, None] * k)
    t = torch.arange(sites, device=center.device, dtype=center.dtype) * width / sites
    basis = torch.exp(1j * omega * t[:, None] * k)
    rotation = torch.exp(
        1j
        * omega
        * torch.tensor([-width, 0.0, width], device=center.device, dtype=center.dtype)[
            :, None
        ]
        * k
    )
    norm = (3 * sites * c.square().sum(-1)).sqrt()
    return owner, beta, basis, rotation, norm


def execute(chart, state, p, x, layout):
    _, centers = coordinates._split(state, chart, p)
    sigma = coordinates.bandwidth_sigma(state, chart, p).detach()
    amp = coordinates.amplitude(state, chart, p)
    output = _axis(chart, centers, sigma, 0, layout[0], state.spec.profiles[0].profile)
    input_ = _axis(chart, centers, sigma, 1, layout[1], state.spec.profiles[1].profile)
    denominator = (output[-1] * input_[-1]).clamp_min(state.spec.normalization.floor)
    scale = denominator.sqrt()
    io, ib, phi_i, rot_i, _ = input_
    oo, ob, phi_o, rot_o, _ = output
    ni, si, _ = layout[1]
    no, so, _ = layout[0]
    # One input transform per tile, shared by all atoms and widths.
    z = x.reshape(x.shape[0], ni, si).to(phi_i.dtype) @ phi_i
    offsets = torch.tensor([-1, 0, 1], device=p.device)
    neighbour_ids = (io[:, None] + offsets).remainder(ni)
    owner_moments = (z[:, neighbour_ids] * rot_i[None, None, :, :]).sum(2)
    h = (owner_moments * ib[None, :, :]).sum(-1).real / scale
    # Aggregate atoms by output owner, then rotate/distribute each group once.
    contributions = (h * (amp / scale)[None, :])[:, :, None] * ob[None, :, :]
    owners = contributions.new_zeros((x.shape[0], no, ob.shape[-1]))
    owners = owners.index_add(1, oo, contributions)
    tile_ids = torch.arange(no, device=p.device)
    distributed = sum(
        owners[:, (tile_ids - n).remainder(no)] * rot_o[n + 1] for n in (-1, 0, 1)
    )
    return (distributed @ phi_o.T).real.reshape(x.shape[0], no * so)
