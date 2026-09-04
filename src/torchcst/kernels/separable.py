"""Separable atom operator kernels."""

from __future__ import annotations

import math

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
        return (
            1
            + self.input_profile.parameter_dim(input_chart)
            + self.output_profile.parameter_dim(output_chart)
        )

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
        amplitude = input_p.new_empty(atoms, 1)
        amplitude.normal_(mean=0.0, std=1.0 / math.sqrt(atoms))
        return torch.cat((amplitude, input_p, output_p), dim=-1)

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
        input_end = 1 + input_dim
        amplitude = p[:, 0]
        phi_input = self.input_profile.evaluate(input_chart, p[:, 1:input_end])
        phi_output = self.output_profile.evaluate(output_chart, p[:, input_end:])
        phi_output = phi_output * amplitude.unsqueeze(0)
        if phi_input.shape[1] != p.shape[0] or phi_output.shape[1] != p.shape[0]:
            raise ValueError("profiles must preserve the atom dimension")
        return phi_input, phi_output

    def extra_repr(self) -> str:
        return f"supports_factorization={self.supports_factorization}"
