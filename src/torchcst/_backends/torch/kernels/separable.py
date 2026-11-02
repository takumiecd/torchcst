"""PyTorch execution for declared separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
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
    return torch.cat((output_p, input_p), dim=-1)


def materialize_atoms(
    state, output_chart: ChartState, input_chart: ChartState, p: Tensor
) -> Tensor:
    phi_input, phi_output = _kernel.factors(state, output_chart, input_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state, output_chart: ChartState, input_chart: ChartState, p: Tensor
) -> tuple[Tensor, Tensor]:
    output_p, input_p = _split(state, output_chart, input_chart, p)
    phi_output = _profile.evaluate(state.profiles[0], output_chart, output_p)
    phi_input = _profile.evaluate(state.profiles[1], input_chart, input_p)
    if phi_input.shape[1] != p.shape[0] or phi_output.shape[1] != p.shape[0]:
        raise ValueError("profiles must preserve the atom dimension")
    return phi_input, phi_output


def tangent_backend(state, output_chart: ChartState, input_chart: ChartState):
    from .tangent import separable

    return lambda p: separable(state, output_chart, input_chart, p)


def _split(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    expected_dim = _kernel.parameter_dim(state, output_chart, input_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    output_dim = _profile.parameter_dim(state.profiles[0], output_chart)
    return p[:, :output_dim], p[:, output_dim:]
