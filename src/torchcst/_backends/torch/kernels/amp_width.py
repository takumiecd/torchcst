"""PyTorch execution for legacy amplitude_bandwidth contracts."""

from __future__ import annotations

import math

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
    input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
    output_p = self.profile.initialize(output_chart, atoms, mode=mode)
    amplitude = input_p.new_empty(atoms, 1)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    return torch.cat((amplitude, input_p, output_p), dim=-1)


def materialize_atoms(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> Tensor:
    phi_input, phi_output = self.factors(input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    amplitude, input_p, output_p = self._split(input_chart, output_chart, p)
    precision, _ = self._precision_and_jacobian(amplitude)
    phi_input = self.profile.evaluate_with_precision(
        input_chart,
        input_p,
        precision,
    )
    phi_output = self.profile.evaluate_with_precision(
        output_chart,
        output_p,
        precision,
    )
    phi_output = phi_output * amplitude.T
    return phi_input, phi_output


def tangent_backend(self, input_chart: Chart, output_chart: Chart):
    from .tangent import amplitude_bandwidth

    return lambda p: amplitude_bandwidth(self, input_chart, output_chart, p)
