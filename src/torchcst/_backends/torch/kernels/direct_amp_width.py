"""PyTorch execution for declared direct_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.parameterizations import (
    direct_amp_width as _parameterizations,
)
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState
from torchcst.kernels.spec import AtomInit
from torchcst.profiling import cst_span


def lower_half_amplitude(state) -> Tensor:
    """Absolute amplitude where ``L(w)`` is halfway between its limits."""

    return state.scalar("w_c") / state.scalar("lower_kappa").sqrt()


def initialize(
    state,
    input_chart: ChartState,
    output_chart: ChartState | int,
    atoms: int | None = None,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    if atoms is None:
        if type(output_chart) is not int:
            raise TypeError("single-chart initialization requires an atom count")
        _kernel.check_radial_activity(state, input_chart)
        atoms = output_chart
        center = _profile.initialize(state.profiles[0], input_chart, atoms, mode=mode)
        amplitude = center.new_empty(atoms).normal_(
            mean=0.0, std=0.1 / math.sqrt(atoms)
        )
        maximum = state.scalar("amplitude_max").to(amplitude)
        amplitude = amplitude.clamp(-maximum, maximum)
        q = torch.ones_like(amplitude) + 3.0 * state.scalar("alpha_init").to(amplitude)
        return torch.cat((torch.stack((amplitude, q), dim=-1), center), dim=-1)
    if not isinstance(output_chart, ChartState):
        raise TypeError("output_chart must be a Chart")
    input_p = _profile.initialize(state.profiles[0], input_chart, atoms, mode="uniform")
    output_p = _profile.initialize(state.profiles[1], output_chart, atoms, mode=mode)

    amplitude = input_p.new_empty(atoms)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    maximum = state.scalar("amplitude_max").to(amplitude)
    amplitude = amplitude.clamp(-maximum, maximum)
    q = torch.ones_like(amplitude) + 3.0 * state.scalar("alpha_init").to(amplitude)
    direct = torch.stack((amplitude, q), dim=-1)
    return torch.cat((direct, input_p, output_p), dim=-1)


def materialize_atoms(
    state,
    input_chart: ChartState,
    output_chart: ChartState | Tensor,
    p: Tensor | None = None,
) -> Tensor:
    if p is None:
        if not isinstance(output_chart, Tensor):
            raise TypeError("single-chart materialization requires atom parameters")
        return _single_materialize_atoms(state, input_chart, output_chart)
    phi_input, phi_output = _kernel.factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    with cst_span("cst.kernel.bandwidth"):
        direct, input_p, output_p = _parameterizations._split(
            state, input_chart, output_chart, p
        )
        amplitude, alpha = _parameterizations._amplitude_and_alpha(state, direct)
        sigma_input, sigma_output = _parameterizations._bandwidth_sigmas(
            state, amplitude, alpha
        )
        # Width is derived from update history, not task-loss gradients.
        precision_input = sigma_input.reciprocal().square().detach()
        precision_output = sigma_output.reciprocal().square().detach()
    with cst_span("cst.kernel.input_profile"):
        phi_input = _profile.evaluate_with_precision(
            state.profiles[0], input_chart, input_p, precision_input
        )
    with cst_span("cst.kernel.output_profile"):
        phi_output = _profile.evaluate_with_precision(
            state.profiles[1], output_chart, output_p, precision_output
        )
    with cst_span("cst.kernel.amplitude_scale"):
        return phi_input, phi_output * amplitude.unsqueeze(0)


def _single_values(
    state,
    chart: ChartState,
    center: Tensor,
    amplitude: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
    values = _profile.evaluate_with_precision_slice(
        state.profiles[0], chart, center, precision, selection
    )
    return values * amplitude.unsqueeze(0)


def _single_block(
    state,
    chart: ChartState,
    center: Tensor,
    amplitude: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
    parts = []
    for start in range(0, center.shape[0], _kernel.options(state).atom_chunk):
        stop = start + _kernel.options(state).atom_chunk
        selected_center = center[start:stop]
        selected_amplitude = amplitude[start:stop]
        selected_precision = precision[start:stop]
        if (
            _kernel.options(state).checkpoint_blocks
            and torch.is_grad_enabled()
            and (selected_center.requires_grad or selected_amplitude.requires_grad)
        ):
            values = checkpoint(
                lambda c, a, prec=selected_precision: _single_values(
                    state, chart, c, a, prec, selection
                ),
                selected_center,
                selected_amplitude,
                use_reentrant=False,
            )
        else:
            values = _single_values(
                state,
                chart,
                selected_center,
                selected_amplitude,
                selected_precision,
                selection,
            )
        parts.append(values.sum(dim=-1))
    return torch.stack(parts).sum(dim=0)


def weight(state, chart: ChartState, p: Tensor) -> Tensor:
    """Sum single-chart atoms with bounded site and atom temporaries."""

    center, amplitude, precision = _parameterizations.tile_parameters(state, chart, p)
    blocks = [
        _single_block(
            state,
            chart,
            center,
            amplitude,
            precision,
            slice(
                start, min(start + _kernel.options(state).site_chunk, chart.features)
            ),
        )
        for start in range(0, chart.features, _kernel.options(state).site_chunk)
    ]
    return torch.cat(blocks).reshape(chart.shape)


def weight_tile(
    state, chart: ChartState, p: Tensor, rows: Tensor, columns: Tensor
) -> Tensor:
    """Sum atom contributions on one [output rows, input columns] tile.

    The tile is a temporary PyTorch tensor. A native tiled GEMM may
    evaluate the same entries directly in registers or shared memory.
    """

    if not hasattr(chart, "shape") or len(chart.shape) != 2:
        raise ValueError("weight_tile requires a two-dimensional operator chart")
    if (
        rows.ndim != 1
        or columns.ndim != 1
        or rows.dtype != torch.long
        or columns.dtype != torch.long
        or rows.device != p.device
        or columns.device != p.device
    ):
        raise ValueError("rows and columns must be long vectors on the atom device")
    if bool(((rows < 0) | (rows >= chart.shape[0])).any()) or bool(
        ((columns < 0) | (columns >= chart.shape[1])).any()
    ):
        raise IndexError("weight tile index out of bounds")
    if p.shape[0] == 0 or rows.numel() == 0 or columns.numel() == 0:
        return p.new_zeros((rows.numel(), columns.numel()))
    center, amplitude, precision = _parameterizations.tile_parameters(state, chart, p)
    sites = (rows[:, None] * chart.shape[1] + columns[None, :]).reshape(-1)
    pieces = [
        _single_block(
            state,
            chart,
            center,
            amplitude,
            precision,
            sites[start : start + _kernel.options(state).site_chunk],
        )
        for start in range(0, sites.numel(), _kernel.options(state).site_chunk)
    ]
    return torch.cat(pieces).reshape(rows.numel(), columns.numel())


def _single_materialize_atoms(state, chart: ChartState, p: Tensor) -> Tensor:
    center, amplitude, precision = _parameterizations.tile_parameters(state, chart, p)
    blocks = [
        _single_values(
            state,
            chart,
            center,
            amplitude,
            precision,
            slice(
                start, min(start + _kernel.options(state).site_chunk, chart.features)
            ),
        )
        for start in range(0, chart.features, _kernel.options(state).site_chunk)
    ]
    return torch.cat(blocks, dim=0).transpose(0, 1).reshape(p.shape[0], *chart.shape)


def packed_weight(state, chart: ChartState, p: Tensor) -> Tensor:
    """Return physically ordered, contiguous tile-major weight storage."""

    if not (isinstance(chart, ChartState) and chart.spec.kind == "strip"):
        raise TypeError("packed_weight requires a StripChart")
    center, amplitude, precision = _parameterizations.tile_parameters(state, chart, p)
    tile_size = math.prod(chart.tile_shape)
    tiles = []
    for station in range(chart.tile_count):
        logical, local = _charts.tile_indices(chart, station)
        values = _single_block(state, chart, center, amplitude, precision, logical)
        tile = values.new_zeros(tile_size).index_copy(0, local, values)
        tiles.append(tile.reshape(chart.tile_shape))
    return torch.stack(tiles)
