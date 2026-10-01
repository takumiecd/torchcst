"""Amplitude composition for complete operator kernels."""

from __future__ import annotations

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
        from torchcst._backends.torch.kernels.amplitude import initialize

        return initialize(self, input_chart, output_chart, atoms, mode=mode)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.kernels.amplitude import materialize_atoms

        return materialize_atoms(self, input_chart, output_chart, p)

    @property
    def supports_factorization(self) -> bool:
        return self.kernel.supports_factorization

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.kernels.amplitude import factors

        return factors(self, input_chart, output_chart, p)

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.kernels.amplitude import _split

        return _split(self, input_chart, output_chart, p)

    def tangent_backend(self, input_chart: Chart, output_chart: Chart):
        from torchcst._backends.torch.kernels.amplitude import tangent_backend

        return tangent_backend(self, input_chart, output_chart)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.updates.amplitude import (
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
        from torchcst._backends.torch.updates.amplitude import apply_parameter_update

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
        from torchcst._backends.torch.updates.amplitude import transport_parameter_state

        return transport_parameter_state(
            self, input_chart, output_chart, old, new, state
        )

    def extra_repr(self) -> str:
        return f"supports_factorization={self.supports_factorization}"
