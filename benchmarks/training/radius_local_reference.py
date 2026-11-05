"""Bounded CPU Fourier factors for radius benchmark initialization/oracles.

This is benchmark tooling, not an execution backend. The public CPU integrated
cell formula and routing law are retained, without allocating [atoms, sites]
factors. Only the dense comparison initializer allocates its required [No, Ni]
output. Its deterministic atom-major scatter has a different FP32 reduction
order from the historical identity/GEMM export.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import pad

from torchcst._backends.torch.kernels.radius_cell_mass import _relative_centers
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values
from torchcst._backends.torch.profiles.radius_fourier import integrated_cells
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes

LOCAL_EXPORT_POLICY = {
    "method": "bounded_public_CPU_Fourier_local_outer_scatter",
    "revision": 1,
    "chunk_atoms": 4096,
    "maximum_outer_product_values": 1048576,
    "factor_dtype": "float32",
    "accumulator_dtype": "float32",
    "reduction": "numpy.add.at; original atom order, then output/input local-site order",
    "graph_scope": "no backward graph during export; no full site-by-atom factors",
}


@dataclass(frozen=True)
class LocalFactors:
    input_sites: torch.Tensor
    input_values: torch.Tensor
    output_sites: torch.Tensor
    output_values: torch.Tensor


def _axis_windows(axis, radii, centers):
    """Public axis_masses routing and cell values, retaining local sites only."""
    if radii.device.type != "cpu" or centers.device.type != "cpu":
        raise ValueError("local reference requires CPU tensors")
    routing = radii.detach().double()
    if not bool(torch.isfinite(routing).all() and torch.isfinite(centers).all()):
        raise ValueError("radius and centers must be finite")
    if bool(((routing <= 0) | (routing > axis.period / 2)).any()):
        raise ValueError("radius must lie in (0, chart period/2]")
    groups = []
    remaining = torch.ones_like(routing, dtype=torch.bool)
    tile_sites = axis.binding.base_tile_sites
    while tile_sites <= axis.site_count:
        tile_width = tile_sites * axis.spacing
        selected = remaining & (routing <= tile_width)
        counts = (2 * routing / tile_width).ceil().long() + 1
        for count in counts[selected].unique().tolist():
            rows = (selected & (counts == count)).nonzero().flatten()
            groups.append((rows, tile_sites, count))
        remaining &= ~selected
        tile_sites *= axis.binding.merge_factor
    if bool(remaining.any()):
        raise ValueError("radius does not fit the declared hierarchy")
    width = max(
        (min(axis.site_count, tile * count) for _, tile, count in groups), default=0
    )
    sites = torch.zeros((len(radii), width), dtype=torch.int64)
    values = radii.new_zeros((len(radii), width), dtype=torch.float64)
    for rows, tile, count in groups:
        local = _relative_centers(centers[rows], axis.origin, axis.period)
        tile_width = tile * axis.spacing
        first = (
            ((local + axis.spacing / 2) / tile_width - (count - 1) / 2).floor().long()
        )
        unwrapped = first[:, None] * tile + torch.arange(count * tile)
        cell_values = integrated_cells(
            axis.binding.profile,
            radii[rows],
            local,
            count * tile_width,
            unwrapped,
            axis.spacing,
        )
        local_sites = unwrapped.remainder(axis.site_count)
        if count * tile > axis.site_count:
            # Match axis_masses: repeated chart images sum in FP64 before the
            # public factors cast to the model dtype, without clipping support.
            cell_values = values.new_zeros((len(rows), axis.site_count)).scatter_add(
                1, local_sites, cell_values
            )
            local_sites = torch.arange(axis.site_count).expand(len(rows), -1)
        padding = width - cell_values.shape[1]
        sites = sites.index_copy(0, rows, pad(local_sites, (0, padding)))
        values = values.index_copy(0, rows, pad(cell_values, (0, padding)))
    return sites, values


def bounded_local_factors(model, parameters=None):
    """Differentiable local public factors for one caller-bounded atom chunk.

    Polar radius/routing is detached, as in the public task VJP. Amplitude and both
    centers remain live. Caller chunking bounds memory and autograd lifetime.
    """
    p = model.atoms.p if parameters is None else parameters
    if p.device.type != "cpu" or p.ndim != 2 or p.shape[1] != 4:
        raise ValueError("local reference requires CPU [atoms,4] parameters")
    if p.dtype not in (torch.float32, torch.float64):
        raise ValueError("local reference requires float32 or float64")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("parameters must be finite")
    output_axis, input_axis = radius_cell_mass_axes(
        tuple(c.declaration() for c in model.cst_charts()), model.kernel.declaration()
    )
    amplitude, ri, ro = decoded_values(model.kernel, p)
    input_sites, input_mass = _axis_windows(input_axis, ri.detach(), p[:, 3])
    output_sites, output_mass = _axis_windows(output_axis, ro.detach(), p[:, 2])
    return LocalFactors(
        input_sites,
        input_mass.to(p.dtype),
        output_sites,
        (output_mass * amplitude[:, None]).to(p.dtype),
    )


def local_reference(model, inputs, parameters=None):
    """One chunk's public Fourier response, with Y/dX/dP autograd support."""
    factors = bounded_local_factors(model, parameters)
    if inputs.device.type != "cpu" or inputs.ndim != 2:
        raise ValueError("local reference requires CPU [batch,sites] inputs")
    if inputs.shape[1] != model.in_features:
        raise ValueError("local reference input width differs from the model")
    response = (inputs[:, factors.input_sites] * factors.input_values[None]).sum(-1)
    contributions = response[:, :, None] * factors.output_values[None]
    return inputs.new_zeros((len(inputs), model.out_features)).scatter_add(
        1,
        factors.output_sites.reshape(1, -1).expand(len(inputs), -1),
        contributions.flatten(1),
    )


def local_dense_weight(
    model, *, chunk_atoms=4096, maximum_outer_product_values=1048576
):
    """Export dense FP32 W outside training, with bounded local scratch.

    numpy.add.at applies repeated additions in specified atom-major order. This
    preserves all contributions and fixes the reduction policy independently of
    Torch's CPU GEMM implementation. No CUDA value or kernel is used as oracle.
    """
    import numpy as np

    if type(chunk_atoms) is not int or chunk_atoms < 1:
        raise ValueError("chunk_atoms must be positive")
    if (
        type(maximum_outer_product_values) is not int
        or maximum_outer_product_values < 1
    ):
        raise ValueError("maximum_outer_product_values must be positive")
    if model.atoms.p.device.type != "cpu" or model.atoms.p.dtype != torch.float32:
        raise ValueError("dense initialization export requires CPU FP32 parameters")
    weight = np.zeros((model.out_features, model.in_features), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, model.atoms.count, chunk_atoms):
            factors = bounded_local_factors(
                model, model.atoms.p[start : start + chunk_atoms]
            )
            ni, no = factors.input_sites.shape[1], factors.output_sites.shape[1]
            if not ni or not no:
                continue
            if ni * no > maximum_outer_product_values:
                raise ValueError(
                    "one atom's local product exceeds the export scratch budget"
                )
            block_atoms = maximum_outer_product_values // (ni * no)
            for offset in range(0, len(factors.input_sites), block_atoms):
                part = slice(offset, offset + block_atoms)
                indices = (
                    factors.output_sites[part, :, None] * model.in_features
                    + factors.input_sites[part, None, :]
                )
                values = (
                    factors.output_values[part, :, None]
                    * factors.input_values[part, None, :]
                )
                np.add.at(
                    weight.reshape(-1),
                    indices.numpy().reshape(-1),
                    values.numpy().reshape(-1),
                )
    return torch.from_numpy(weight)
