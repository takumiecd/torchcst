"""PyTorch execution for legacy separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart
from torchcst.kernels.base import AtomInit


def initialize(
    self,
    input_chart: Chart,
    output_chart: Chart,
    atoms: int,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    input_p = self.input_profile.initialize(input_chart, atoms, mode="uniform")
    output_p = self.output_profile.initialize(output_chart, atoms, mode=mode)
    return torch.cat((input_p, output_p), dim=-1)


def materialize_atoms(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    phi_input, phi_output = self.factors(input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    expected_dim = self.parameter_dim(input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = self.input_profile.parameter_dim(input_chart)
    phi_input = self.input_profile.evaluate(input_chart, p[:, :input_dim])
    phi_output = self.output_profile.evaluate(output_chart, p[:, input_dim:])
    if phi_input.shape[1] != p.shape[0] or phi_output.shape[1] != p.shape[0]:
        raise ValueError("profiles must preserve the atom dimension")
    return phi_input, phi_output


def tangent_backend(self, input_chart: Chart, output_chart: Chart):
    if not (
        self.input_profile.supports_tangent and self.output_profile.supports_tangent
    ):
        return None
    from .tangent import separable

    return lambda p: separable(self, input_chart, output_chart, p)


def _split(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    expected_dim = self.parameter_dim(input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = self.input_profile.parameter_dim(input_chart)
    return p[:, :input_dim], p[:, input_dim:]
