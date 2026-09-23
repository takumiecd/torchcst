"""Amplitude composition for complete operator kernels."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel


class Amplitude(Kernel):
    r"""Prepend a signed scalar amplitude and evaluate ``w * kernel(q)``."""

    def __init__(self, kernel: Kernel) -> None:
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must implement the Kernel contract")
        self.kernel = kernel

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return 1 + self.kernel.parameter_dim(input_chart, output_chart)

    def parameter_dof(self, input_chart: Chart, output_chart: Chart) -> int:
        return 1 + self.kernel.parameter_dof(input_chart, output_chart)

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

    @property
    def supports_factorization(self) -> bool:
        return self.kernel.supports_factorization

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.supports_factorization:
            return super().factors(input_chart, output_chart, p)
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
        from ._tangent import amplitude

        return lambda p: amplitude(self, inner, input_chart, output_chart, p)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        _, inner = self._split(input_chart, output_chart, p)
        amplitude_gradient, inner_gradient = self._split(
            input_chart, output_chart, gradient
        )
        projected = self.kernel.project_parameter_gradient(
            input_chart,
            output_chart,
            inner,
            inner_gradient,
        )
        return torch.cat((amplitude_gradient, projected), dim=-1)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        amplitude, inner = self._split(input_chart, output_chart, p)
        amplitude_delta, inner_delta = self._split(
            input_chart, output_chart, displacement
        )
        updated = self.kernel.apply_parameter_update(
            input_chart,
            output_chart,
            inner,
            inner_delta,
            step_size=step_size,
        )
        return torch.cat((amplitude + amplitude_delta, updated), dim=-1)

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        _, old_inner = self._split(input_chart, output_chart, old)
        _, new_inner = self._split(input_chart, output_chart, new)
        amplitude_state, inner_state = self._split(input_chart, output_chart, state)
        transported = self.kernel.transport_parameter_state(
            input_chart,
            output_chart,
            old_inner,
            new_inner,
            inner_state,
        )
        return torch.cat((amplitude_state, transported), dim=-1)

    def extra_repr(self) -> str:
        return f"supports_factorization={self.supports_factorization}"
