"""PyTorch execution for declared direct_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.parameterizations import (
    direct_amp_width as _parameterizations,
)
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import ChartState
from torchcst.kernels.spec import AtomInit
from torchcst.profiling import cst_span


def lower_half_amplitude(state) -> Tensor:
    """Absolute amplitude where ``L(w)`` is halfway between its limits."""

    return state.scalar("w_c") / state.scalar("lower_kappa").sqrt()


def initialize(
    state,
    output_chart: ChartState,
    input_chart: ChartState | int,
    atoms: int | None = None,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    if atoms is None:
        if type(input_chart) is not int:
            raise TypeError("single-chart initialization requires an atom count")
        _kernel.check_radial_activity(state, output_chart)
        atoms = input_chart
        center = _profile.initialize(state.profiles[0], output_chart, atoms, mode=mode)
        amplitude = center.new_empty(atoms).normal_(
            mean=0.0, std=0.1 / math.sqrt(atoms)
        )
        maximum = state.scalar("amplitude_max").to(amplitude)
        amplitude = amplitude.clamp(-maximum, maximum)
        q = torch.ones_like(amplitude) + 3.0 * state.scalar("alpha_init").to(amplitude)
        return torch.cat((torch.stack((amplitude, q), dim=-1), center), dim=-1)
    if not isinstance(input_chart, ChartState):
        raise TypeError("input_chart must be a Chart")
    output_p = _profile.initialize(state.profiles[0], output_chart, atoms, mode=mode)
    input_p = _profile.initialize(
        state.profiles[-1], input_chart, atoms, mode="uniform"
    )

    amplitude = output_p.new_empty(atoms)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    maximum = state.scalar("amplitude_max").to(amplitude)
    amplitude = amplitude.clamp(-maximum, maximum)
    q = torch.ones_like(amplitude) + 3.0 * state.scalar("alpha_init").to(amplitude)
    direct = torch.stack((amplitude, q), dim=-1)
    return torch.cat((direct, output_p, input_p), dim=-1)


def materialize_atoms(
    state,
    output_chart: ChartState,
    input_chart: ChartState | Tensor,
    p: Tensor | None = None,
) -> Tensor:
    if p is None:
        if not isinstance(input_chart, Tensor):
            raise TypeError("single-chart materialization requires atom parameters")
        return _single_materialize_atoms(state, output_chart, input_chart)
    phi_input, phi_output = _kernel.factors(state, output_chart, input_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    with cst_span("cst.kernel.bandwidth"):
        direct, output_p, input_p = _parameterizations._split(
            state, output_chart, input_chart, p
        )
        amplitude, alpha = _parameterizations._amplitude_and_alpha(state, direct)
        sigma_input, sigma_output = _parameterizations._bandwidth_sigmas(
            state, amplitude, alpha
        )
        # Width is derived from update history, not task-loss gradients.
        precision_input = sigma_input.reciprocal().square().detach()
        precision_output = sigma_output.reciprocal().square().detach()
    with cst_span("cst.kernel.output_profile"):
        phi_output = _profile.evaluate_with_precision(
            state.profiles[0], output_chart, output_p, precision_output
        )
    with cst_span("cst.kernel.input_profile"):
        phi_input = _profile.evaluate_with_precision(
            state.profiles[-1], input_chart, input_p, precision_input
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
