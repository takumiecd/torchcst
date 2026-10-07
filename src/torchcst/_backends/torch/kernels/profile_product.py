"""Reference axis contractions for a single Euclidean Product/Strip chart."""

import math

import torch

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.parameterizations import profile_product as _coordinates
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import StripChartState


def initialize(state, chart, atoms, *, mode):
    centers = _charts.initialize_centers(chart, atoms, mode=mode)
    amplitude = centers.new_empty(atoms).normal_(0.0, 0.1 / math.sqrt(atoms))
    ratio = (amplitude / state.scalar("amplitude_max").to(centers)).clamp(
        -1 + 1e-6, 1 - 1e-6
    )
    polar = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
    radius = (1 + 3 * state.scalar("alpha_init").to(polar)).sqrt()
    return torch.cat((polar * radius, centers), dim=-1)


def lower_half_amplitude(state):
    return state.scalar("w_c") / state.scalar("lower_kappa").sqrt()


def _axis_positions(chart):
    """Independent vectors, including Strip's physical pitch and partial end."""
    for logical_axis, pattern in enumerate(chart.axes):
        for local_axis, size in enumerate(pattern.spec.shape):
            indices = torch.arange(size, device=chart.device)
            if isinstance(chart, StripChartState) and logical_axis == chart.axis:
                tile = chart.tile_shape[logical_axis]
                yield (
                    pattern.start[local_axis]
                    + (indices % tile).to(chart.dtype) * pattern.spacing[local_axis]
                    + torch.div(indices, tile, rounding_mode="floor").to(chart.dtype)
                    * chart.tile_pitch
                )
            else:
                yield (
                    pattern.start[local_axis]
                    + indices.to(chart.dtype) * pattern.spacing[local_axis]
                )


def _product(values):
    """Flatten a Cartesian product with the last coordinate varying fastest."""
    result = values[0]
    for value in values[1:]:
        result = (result[:, None, :] * value[None, :, :]).flatten(0, 1)
    return result


def factors(state, chart, p):
    _, centers = _coordinates._split(state, chart, p)
    amplitude = _coordinates.amplitude(state, chart, p)
    # Preserve Polar's task derivative: widths evolve through the update law,
    # but their current values are fixed in the task VJP.
    precision = _coordinates.bandwidth_precision(state, chart, p).detach()
    axes = [
        _profile.shape_function(
            profile,
            "unnormalized_from_squared",
            (positions[:, None] - centers[None, :, dim]).square(),
            precision,
        )
        for dim, (positions, profile) in enumerate(
            zip(_axis_positions(chart), state.profiles)
        )
    ]
    # The full-domain norm factors because the grid is Cartesian. Floor once,
    # after the product; independently flooring axes changes empty/tiny support.
    norm_dtype = (
        torch.float32 if p.dtype in (torch.float16, torch.bfloat16) else p.dtype
    )
    norm = torch.ones_like(amplitude, dtype=norm_dtype)
    for value in axes:
        norm = norm * torch.linalg.vector_norm(value.to(norm_dtype), dim=0)
    denominator = norm.clamp_min(state.spec.normalization.floor)
    output_dims = chart.axes[0].spec.dim
    # Split the denominator so half precision does not form amp/floor > 65504
    # and then multiply an empty profile by infinity. The complete product and
    # one global floor are unchanged. Keep norms in FP32 for low-precision inputs.
    scale = denominator.sqrt()[None, :]
    phi_output = ((_product(axes[:output_dims]) / scale) * amplitude[None, :]).to(
        p.dtype
    )
    phi_input = (_product(axes[output_dims:]) / scale).to(p.dtype)
    return phi_input, phi_output


def materialize_atoms(state, chart, p):
    phi_input, phi_output = factors(state, chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)
