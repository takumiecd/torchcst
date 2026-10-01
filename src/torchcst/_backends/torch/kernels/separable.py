"""PyTorch execution for declared separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry import Chart
from torchcst.kernels.spec import AtomInit


def initialize(
    state,
    input_chart: Chart,
    output_chart: Chart,
    atoms: int,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    input_p = _profile.initialize(state.profiles[0], input_chart, atoms, mode="uniform")
    output_p = _profile.initialize(state.profiles[1], output_chart, atoms, mode=mode)
    return torch.cat((input_p, output_p), dim=-1)


def materialize_atoms(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    phi_input, phi_output = _kernel.factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    expected_dim = _kernel.parameter_dim(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = _profile.parameter_dim(state.profiles[0], input_chart)
    phi_input = _profile.evaluate(state.profiles[0], input_chart, p[:, :input_dim])
    phi_output = _profile.evaluate(state.profiles[1], output_chart, p[:, input_dim:])
    if phi_input.shape[1] != p.shape[0] or phi_output.shape[1] != p.shape[0]:
        raise ValueError("profiles must preserve the atom dimension")
    return phi_input, phi_output


def tangent_backend(state, input_chart: Chart, output_chart: Chart):
    from .tangent import separable

    return lambda p: separable(state, input_chart, output_chart, p)


def _split(
    state,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    expected_dim = _kernel.parameter_dim(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = _profile.parameter_dim(state.profiles[0], input_chart)
    return p[:, :input_dim], p[:, input_dim:]
