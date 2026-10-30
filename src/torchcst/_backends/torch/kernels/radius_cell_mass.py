"""Radius-native CPU cell masses and complete atom responses.

The reference uses direct probability factors. It does not allocate hierarchical
moments or expose partial responses. Routing is piecewise constant in backward.
"""

import torch

from torchcst.kernels.parameterizations import PhysicalAmpRadiusSpec
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes

from ..profiles.radius_fourier import integrated_cells


def axis_masses(axis, radius, centers):
    """Select a hierarchy and retain exactly one padded local Fourier period.

    Even and odd K use the nearest tile-aligned K-tile window. Integer ties go
    to the higher first tile. Cells wrap onto the chart; if P exceeds Q, repeated
    chart images are added rather than clipping P or renormalizing chart cells.
    """
    if radius.device.type != "cpu" or centers.device.type != "cpu":
        raise ValueError("radius cell-mass reference currently requires CPU tensors")
    if radius.ndim != 1 or centers.shape != radius.shape:
        raise ValueError("radius and centers must have equal vector shapes")
    if not bool(torch.isfinite(centers).all()) or not bool(
        torch.isfinite(radius).all()
    ):
        raise ValueError("radius and centers must be finite")
    # Compare the encoded radii with the physical FP64 chart thresholds. A
    # Python float compared to FP32 would round the threshold to FP32 first.
    # Only routing is detached; integrated_cells below keeps the live radius.
    routing_radius = radius.detach().double()
    if bool(((routing_radius <= 0) | (routing_radius > axis.period / 2)).any()):
        raise ValueError("radius must lie in (0, chart period/2]")
    result = (radius.double()[:, None] + centers.double()[:, None]).expand(
        -1, axis.site_count
    ) * 0
    remaining = torch.ones_like(radius, dtype=torch.bool)
    tile_sites = axis.binding.base_tile_sites
    while tile_sites <= axis.site_count:
        tile_width = tile_sites * axis.spacing
        selected = remaining & (routing_radius <= tile_width)
        # Ceil and integer routing are deliberately outside the smooth branch.
        counts = (2 * routing_radius / tile_width).ceil().long() + 1
        for count in counts[selected].unique().tolist():
            indices = (selected & (counts == count)).nonzero().flatten()
            local_center = (centers[indices].double() - axis.origin).remainder(
                axis.period
            )
            first_tile = (
                ((local_center + axis.spacing / 2) / tile_width - (count - 1) / 2)
                .floor()
                .long()
            )
            offsets = torch.arange(count * tile_sites)
            sites = first_tile[:, None] * tile_sites + offsets
            values = integrated_cells(
                axis.binding.profile,
                radius[indices],
                local_center,
                count * tile_width,
                sites,
                axis.spacing,
            )
            masses = result.new_zeros((indices.numel(), axis.site_count)).scatter_add(
                1, sites.remainder(axis.site_count), values
            )
            result = result.index_copy(0, indices, masses)
        remaining = remaining & ~selected
        tile_sites *= axis.binding.merge_factor
    if bool(remaining.any()):
        raise ValueError("radius does not fit the declared hierarchy")
    return result


def initialize(state, input_chart, output_chart, atoms, *, mode):
    if input_chart.reference.device.type != "cpu":
        raise ValueError("radius cell-mass reference currently requires CPU tensors")
    axes = radius_cell_mass_axes(
        (input_chart.declaration(), output_chart.declaration()), state.declaration()
    )
    if atoms == 0:
        return input_chart.reference.new_empty((0, 4))
    if type(state.spec.parameterization) is not PhysicalAmpRadiusSpec:
        from .polar_amp_width import initialize as initialize_polar

        return initialize_polar(state, input_chart, output_chart, atoms, mode=mode)
    low, high = (
        float(state.scalar(name)) for name in ("radius_init_min", "radius_init_max")
    )
    p = input_chart.reference.new_empty((atoms, 4))
    p[:, 0].normal_(0, float(state.scalar("amplitude_std")))
    p[:, 1].uniform_(low, high)
    p[:, 2].uniform_(axes[1].origin, axes[1].origin + axes[1].period)
    p[:, 3].uniform_(axes[0].origin, axes[0].origin + axes[0].period)
    return p


def factors(state, input_chart, output_chart, p):
    if p.device.type != "cpu":
        raise ValueError("radius cell-mass reference currently requires CPU tensors")
    if p.ndim != 2 or p.shape[1] != 4 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError("radius cell masses require floating-point [atom,4] rows")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("atom parameters must be finite")
    axes = radius_cell_mass_axes(
        (input_chart.declaration(), output_chart.declaration()), state.declaration()
    )
    if type(state.spec.parameterization) is PhysicalAmpRadiusSpec:
        amplitude = p[:, 0].double()
        input_radius = output_radius = p[:, 1]
        input_center, output_center = p[:, 3], p[:, 2]
    else:
        from ..parameterizations.radius_cell_mass import decoded_values

        amplitude, input_radius, output_radius = decoded_values(state, p)
        input_radius, output_radius = input_radius.detach(), output_radius.detach()
        input_center, output_center = p[:, 2], p[:, 3]
    input_mass = axis_masses(axes[0], input_radius, input_center)
    output_mass = axis_masses(axes[1], output_radius, output_center)
    return input_mass.T.to(p.dtype), (output_mass.T * amplitude).to(p.dtype)


def materialize_atoms(state, input_chart, output_chart, p):
    inputs, outputs = factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", outputs, inputs)


def weight(state, input_chart, output_chart, p):
    """Dense export in chunks; forward uses complete H through Torch factored."""
    result = p.new_zeros((output_chart.features, input_chart.features))
    for start in range(0, len(p), 128):
        inputs, outputs = factors(
            state, input_chart, output_chart, p[start : start + 128]
        )
        result = result + outputs @ inputs.T
    return result
