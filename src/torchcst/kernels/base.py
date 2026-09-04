"""Contracts for atom operator kernels and their scalar profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor, nn

from torchcst.geometry import Chart

AtomInit = Literal["balanced", "uniform"]


class Profile(nn.Module, ABC):
    """A fixed scalar profile that interprets one slice of an atom coordinate."""

    @abstractmethod
    def parameter_dim(self, chart: Chart) -> int:
        """Return the number of opaque coordinates consumed for one atom."""

    @abstractmethod
    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        """Create an ``[atoms, parameter_dim]`` coordinate table."""

    @abstractmethod
    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        """Return profile values with shape ``[chart.features, atoms]``."""


class Kernel(nn.Module, ABC):
    """Interpret each opaque atom row as a complete operator contribution."""

    @abstractmethod
    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        """Return the opaque coordinate width ``P`` for one atom."""

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

    def forward(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
        return self.materialize_atoms(input_chart, output_chart, p)
