"""Differentiable cell-mass reference for the public physical-atom contract.

This independent reference integrates Fourier cells directly. It does not call
CUDA Z/H/C code, and is not the CUDA runtime's coefficient storage policy.
"""

import torch

from torchcst.kernels.cell_mass import CellMassAxisSpec
from torchcst.kernels.cell_mass_layout import cell_mass_axes

from .cell_mass import profile_cell_masses


def _axes(state, input_chart, output_chart):
    return cell_mass_axes(
        (input_chart.declaration(), output_chart.declaration()), state.declaration()
    )


def initialize(state, input_chart, output_chart, atoms, *, mode):
    axes = _axes(state, input_chart, output_chart)
    low, high = (
        float(state.scalar(name)) for name in ("sigma_init_min", "sigma_init_max")
    )
    std = float(state.scalar("amplitude_std"))
    if not 0 < low <= high <= min(a.period for a in axes) / 3 or std < 0:
        raise ValueError("cell-mass initialization must fit the physical sigma domain")
    p = input_chart.reference.new_empty((atoms, 4))
    p[:, 0].normal_(0, std)
    p[:, 1].uniform_(low, high)
    # Physical storage follows the qualified CUDA contract: output, then input.
    p[:, 2].uniform_(axes[1].origin, axes[1].origin + axes[1].period)
    p[:, 3].uniform_(axes[0].origin, axes[0].origin + axes[0].period)
    return p


def axis_masses(axis, sigma, centers):
    """Route each atom once, then integrate its selected one-period cells."""
    result = (sigma[:, None] + centers[:, None]).expand(
        -1, axis.site_count
    ).double() * 0
    remaining = torch.ones_like(sigma, dtype=torch.bool)
    tile = axis.binding.min_tile_sites
    while 3 * tile <= axis.site_count:
        selected = remaining & (sigma <= tile * axis.spacing)
        indices = selected.nonzero().flatten()
        spec = CellMassAxisSpec(
            profile=axis.binding.profile,
            site_count=axis.site_count,
            spacing=axis.spacing,
            origin=axis.origin,
            tile_sites=tile,
        )
        values = profile_cell_masses(spec, centers[indices], sigma[indices])
        result = result.index_copy(0, indices, values)
        remaining = remaining & ~selected
        tile *= 2
    indices = remaining.nonzero().flatten()
    spec = CellMassAxisSpec(
        profile=axis.binding.profile,
        site_count=axis.site_count,
        spacing=axis.spacing,
        origin=axis.origin,
        tile_sites=axis.site_count,
        period_tiles=1,
    )
    values = profile_cell_masses(spec, centers[indices], sigma[indices])
    return result.index_copy(0, indices, values)


def factors(state, input_chart, output_chart, p):
    axes = _axes(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != 4 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("cell masses require FP32/FP64 physical [atom,4] parameters")
    if not bool(torch.isfinite(p).all()) or bool(
        ((p[:, 1] <= 0) | (p[:, 1] > min(a.period for a in axes) / 3)).any()
    ):
        raise ValueError("invalid physical cell-mass atom or sigma")
    input_mass = axis_masses(axes[0], p[:, 1], p[:, 3])
    output_mass = axis_masses(axes[1], p[:, 1], p[:, 2])
    return input_mass.T.to(p.dtype), (output_mass.T * p[:, 0]).to(p.dtype)


def materialize_atoms(state, input_chart, output_chart, p):
    v, u = factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", u, v)


def weight(state, input_chart, output_chart, p):
    """Explicit dense export without an [atom,output,input] temporary."""
    result = p.new_zeros((output_chart.features, input_chart.features))
    for start in range(0, len(p), 512):
        v, u = factors(state, input_chart, output_chart, p[start : start + 512])
        result = result + u @ v.T
    return result
