"""PyTorch execution for declared amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.parameterizations import amp_width as _parameterizations
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import ChartState
from torchcst.kernels.spec import AtomInit


def initialize(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    atoms: int,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    output_p = _profile.initialize(state.profiles[0], output_chart, atoms, mode=mode)
    input_p = _profile.initialize(state.profiles[1], input_chart, atoms, mode="uniform")
    amplitude = output_p.new_empty(atoms, 1)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    return torch.cat((amplitude, output_p, input_p), dim=-1)


def materialize_atoms(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
) -> Tensor:
    phi_input, phi_output = _kernel.factors(state, output_chart, input_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    amplitude, output_p, input_p = _parameterizations._split(
        state, output_chart, input_chart, p
    )
    precision, _ = _parameterizations._precision_and_jacobian(state, amplitude)
    phi_output = _profile.evaluate_with_precision(
        state.profiles[0], output_chart, output_p, precision
    )
    phi_input = _profile.evaluate_with_precision(
        state.profiles[1], input_chart, input_p, precision
    )
    phi_output = phi_output * amplitude.T
    return phi_input, phi_output


def tangent_backend(state, output_chart: ChartState, input_chart: ChartState):
    from .tangent import amplitude_bandwidth

    return lambda p: amplitude_bandwidth(state, output_chart, input_chart, p)
