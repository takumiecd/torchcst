"""Contracts for atom operator kernels and their scalar profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn

from torchcst.geometry import Chart

AtomInit = Literal["balanced", "uniform"]


class Profile(nn.Module, ABC):
    """A fixed scalar profile that interprets one slice of an atom coordinate."""

    def get_extra_state(self) -> dict[str, object]:
        """Record the profile type and non-buffer evaluation settings."""

        return {
            "format_version": 1,
            "profile_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "tangent_config": self.tangent_config(),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("profile checkpoint contract differs from this profile")

    @abstractmethod
    def parameter_dim(self, chart: Chart) -> int:
        """Return the number of opaque coordinates consumed for one atom."""

    def parameter_dof(self, chart: Chart) -> int:
        """Return the geometric degrees of freedom consumed for one atom."""

        return self.parameter_dim(chart)

    @abstractmethod
    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        """Create an ``[atoms, parameter_dim]`` coordinate table."""

    @abstractmethod
    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        """Return profile values with shape ``[chart.features, atoms]``."""

    @property
    def supports_tangent(self) -> bool:
        return False

    def tangent(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        """Optional values and first coordinate derivatives [features, atoms, q]."""
        raise NotImplementedError("profile has no specialized tangent")

    def project_gradient(self, chart: Chart, p: Tensor, gradient: Tensor) -> Tensor:
        """Project a coordinate gradient into the profile's tangent space."""

        del chart, p
        if gradient.ndim != 2:
            raise ValueError("profile gradient must have shape [atoms, parameters]")
        return gradient

    def apply_parameter_update(
        self,
        chart: Chart,
        p: Tensor,
        displacement: Tensor,
    ) -> Tensor:
        """Apply one proposal in the profile coordinate geometry."""

        del chart
        if p.shape != displacement.shape:
            raise ValueError("p and displacement must have matching shapes")
        return p + displacement

    def transport_state(
        self,
        chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Transport a vector-like optimizer state between profile points."""

        del chart
        if old.shape != new.shape or old.shape != state.shape:
            raise ValueError("old, new, and state must have matching shapes")
        return state

    def tangent_config(self) -> tuple:
        """Fixed non-buffer settings affecting evaluation, for custom profiles."""
        return ()


class Kernel(nn.Module, ABC):
    """Interpret each opaque atom row as a complete operator contribution."""

    def get_extra_state(self) -> dict[str, object]:
        """Record the kernel type and fixed non-buffer settings."""

        return {
            "format_version": 1,
            "kernel_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "tangent_config": self.tangent_config(),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("kernel checkpoint contract differs from this kernel")

    @abstractmethod
    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        """Return the opaque coordinate width ``P`` for one atom."""

    def parameter_dof(self, input_chart: Chart, output_chart: Chart) -> int:
        """Return the geometric degrees of freedom in one atom row."""

        return self.parameter_dim(input_chart, output_chart)

    @abstractmethod
    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: AtomInit,
    ) -> Tensor:
        """Create an opaque coordinate table with shape ``[atoms, P]``."""

    @abstractmethod
    def materialize_atoms(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Evaluate ``Kernel(p[a])`` as one matrix per atom.

        The result has shape ``[K, out_features, in_features]``. Every
        amplitude, location, bandwidth, and other trainable atom property is
        encoded in ``p`` and interpreted here.
        """

    @property
    def supports_factorization(self) -> bool:
        """Whether ``factors`` supplies an exact execution factorization."""

        return False

    def factors(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return factors whose outer products equal ``Kernel(p[a])`` exactly."""

        raise NotImplementedError("this kernel does not provide a factorized backend")

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        """Project a parameter gradient into the kernel parameter tangent space."""

        del input_chart, output_chart, p
        if gradient.ndim != 2:
            raise ValueError("kernel gradient must have shape [atoms, parameters]")
        return gradient

    def forward(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
        return self.materialize_atoms(input_chart, output_chart, p)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        """Apply one optimizer proposal in this kernel's coordinate geometry.

        Ordinary kernels use Euclidean addition. Kernels with redundant or
        constrained coordinates may project the task displacement, apply a
        decoupled regularizer, and retract onto their feasible set here. The
        optimizer deliberately does not interpret columns of the opaque atom
        table.
        """

        del input_chart, output_chart, step_size
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        return p + displacement

    def transport_parameter_state(
        self,
        input_chart: Chart,
        output_chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Transport vector-like optimizer state after a constrained update."""

        del input_chart, output_chart
        if old.shape != new.shape or old.shape != state.shape:
            raise ValueError("old, new, and state must have matching shapes")
        return state

    tangent_layout_version = 1

    def tangent_config(self) -> tuple:
        """Fixed non-buffer tangent settings; custom kernels must include them.

        Tensor settings belong in registered buffers. Nested modules/buffers
        are recorded separately. No optimizer history belongs here.
        """
        return ()

    def tangent_backend(self, input_chart: Chart, output_chart: Chart):
        """Optional point -> (U, V, dU, dV), in atom-major order.

        None selects exact factor-autograd or reference fallback.
        """
