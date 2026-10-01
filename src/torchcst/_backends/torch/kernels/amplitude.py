"""PyTorch execution for legacy amplitude contracts."""

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
    inner = self.kernel.initialize(
        input_chart,
        output_chart,
        atoms,
        mode=mode,
    )
    amplitude = inner.new_empty(atoms, 1)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    return torch.cat((amplitude, inner), dim=-1)


def materialize_atoms(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> Tensor:
    amplitude, inner = self._split(input_chart, output_chart, p)
    represented = self.kernel.materialize_atoms(
        input_chart,
        output_chart,
        inner,
    )
    return amplitude[:, None] * represented


def factors(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    if not self.supports_factorization:
        from torchcst.kernels.amplitude import Amplitude

        return super(Amplitude, self).factors(input_chart, output_chart, p)
    amplitude, inner = self._split(input_chart, output_chart, p)
    phi_input, phi_output = self.kernel.factors(
        input_chart,
        output_chart,
        inner,
    )
    return phi_input, phi_output * amplitude.T


def _split(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    expected_dim = self.parameter_dim(input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    return p[:, :1], p[:, 1:]


def tangent_backend(self, input_chart: Chart, output_chart: Chart):
    inner = self.kernel.tangent_backend(input_chart, output_chart)
    if inner is None:
        return None
    from .tangent import amplitude

    return lambda p: amplitude(self, inner, input_chart, output_chart, p)
