"""Separable atom operator kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile


class Separable(Kernel):
    """Compose input and output scalar profiles into rank-one operator atoms."""

    def __init__(self, *, input_profile: Profile, output_profile: Profile) -> None:
        super().__init__()
        if not isinstance(input_profile, Profile) or not isinstance(
            output_profile, Profile
        ):
            raise TypeError(
                "input_profile and output_profile must be Profile instances"
            )
        self.input_profile = input_profile
        self.output_profile = output_profile

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return self.input_profile.parameter_dim(
            input_chart
        ) + self.output_profile.parameter_dim(output_chart)

    def parameter_dof(self, input_chart: Chart, output_chart: Chart) -> int:
        return self.input_profile.parameter_dof(
            input_chart
        ) + self.output_profile.parameter_dof(output_chart)

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

    @property
    def supports_factorization(self) -> bool:
        return True

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
        from ._tangent import separable

        return lambda p: separable(self, input_chart, output_chart, p)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        input_p, output_p = self._split(input_chart, output_chart, p)
        input_g, output_g = self._split(input_chart, output_chart, gradient)
        return torch.cat(
            (
                self.input_profile.project_gradient(input_chart, input_p, input_g),
                self.output_profile.project_gradient(output_chart, output_p, output_g),
            ),
            dim=-1,
        )

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        del step_size
        input_p, output_p = self._split(input_chart, output_chart, p)
        input_d, output_d = self._split(input_chart, output_chart, displacement)
        return torch.cat(
            (
                self.input_profile.apply_parameter_update(
                    input_chart, input_p, input_d
                ),
                self.output_profile.apply_parameter_update(
                    output_chart, output_p, output_d
                ),
            ),
            dim=-1,
        )

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        old_i, old_o = self._split(input_chart, output_chart, old)
        new_i, new_o = self._split(input_chart, output_chart, new)
        state_i, state_o = self._split(input_chart, output_chart, state)
        return torch.cat(
            (
                self.input_profile.transport_state(input_chart, old_i, new_i, state_i),
                self.output_profile.transport_state(
                    output_chart, old_o, new_o, state_o
                ),
            ),
            dim=-1,
        )

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

    def extra_repr(self) -> str:
        return f"supports_factorization={self.supports_factorization}"
