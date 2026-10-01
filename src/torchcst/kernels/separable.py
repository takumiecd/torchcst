"""Separable atom operator kernels."""

from __future__ import annotations

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
        from torchcst._backends.torch.kernels.separable import initialize

        return initialize(self, input_chart, output_chart, atoms, mode=mode)

    def materialize_atoms(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.kernels.separable import materialize_atoms

        return materialize_atoms(self, input_chart, output_chart, p)

    @property
    def supports_factorization(self) -> bool:
        return True

    def factors(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.kernels.separable import factors

        return factors(self, input_chart, output_chart, p)

    def tangent_backend(self, input_chart: Chart, output_chart: Chart):
        from torchcst._backends.torch.kernels.separable import tangent_backend

        return tangent_backend(self, input_chart, output_chart)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.separable import (
            project_parameter_gradient,
        )

        return project_parameter_gradient(self, input_chart, output_chart, p, gradient)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        from torchcst._backends.torch.updates.separable import apply_parameter_update

        return apply_parameter_update(
            self, input_chart, output_chart, p, displacement, step_size=step_size
        )

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.separable import transport_parameter_state

        return transport_parameter_state(
            self, input_chart, output_chart, old, new, state
        )

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.kernels.separable import _split

        return _split(self, input_chart, output_chart, p)

    def extra_repr(self) -> str:
        return f"supports_factorization={self.supports_factorization}"
