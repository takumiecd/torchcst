"""A fixed-shape CST linear module."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchcst.geometry import Chart
from torchcst.kernels import Kernel

Backend = Literal["auto", "factored", "materialized"]
AtomInit = Literal["balanced", "uniform"]


class CSTLinear(nn.Module):
    """Represent a linear map with a fixed number of continuous atoms."""

    def __init__(
        self,
        input_chart: Chart,
        output_chart: Chart,
        *,
        atoms: int,
        input_kernel: Kernel,
        output_kernel: Kernel | None = None,
        atom_init: AtomInit = "balanced",
        backend: Backend = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(input_chart, Chart) or not isinstance(output_chart, Chart):
            raise TypeError("input_chart and output_chart must be Chart instances")
        if isinstance(atoms, bool) or not isinstance(atoms, int):
            raise TypeError("atoms must be an integer")
        if atoms < 1:
            raise ValueError("atoms must be positive")
        if not isinstance(input_kernel, Kernel):
            raise TypeError("input_kernel must implement the Kernel contract")
        if output_kernel is not None and not isinstance(output_kernel, Kernel):
            raise TypeError("output_kernel must implement the Kernel contract")
        if atom_init not in ("balanced", "uniform"):
            raise ValueError("atom_init must be 'balanced' or 'uniform'")
        if backend not in ("auto", "factored", "materialized"):
            raise ValueError("backend must be 'auto', 'factored', or 'materialized'")

        selected_output_kernel = output_kernel or input_kernel
        if not input_kernel.supports_dimension(input_chart.dim):
            raise ValueError("input_kernel does not support the input chart dimension")
        if not selected_output_kernel.supports_dimension(output_chart.dim):
            if output_kernel is None:
                raise ValueError(
                    "input_kernel cannot be reused for the output chart dimension"
                )
            raise ValueError("output_kernel does not support the output chart dimension")

        self.input_chart = input_chart
        self.output_chart = output_chart
        self.input_kernel = input_kernel
        self.output_kernel = selected_output_kernel
        self.atoms = atoms
        self.atom_init = atom_init
        self.backend = backend

        target_device = device or input_chart.coordinates.device
        target_dtype = dtype or input_chart.coordinates.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("CSTLinear requires a floating-point dtype")
        self.to(device=target_device, dtype=target_dtype)

        factory_kwargs = {"device": target_device, "dtype": target_dtype}
        self.source = nn.Parameter(torch.empty(atoms, input_chart.dim, **factory_kwargs))
        self.target = nn.Parameter(torch.empty(atoms, output_chart.dim, **factory_kwargs))
        self.amplitude = nn.Parameter(torch.empty(atoms, **factory_kwargs))
        self.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.input_chart.features

    @property
    def out_features(self) -> int:
        return self.output_chart.features

    def reset_parameters(self) -> None:
        """Initialize the fixed atom table without changing its cardinality."""

        with torch.no_grad():
            self.source.copy_(self._sample_bounds(self.input_chart.coordinates))
            if self.atom_init == "balanced":
                indices = torch.linspace(
                    0,
                    self.out_features - 1,
                    self.atoms,
                    device=self.target.device,
                ).round().to(dtype=torch.long)
                self.target.copy_(self.output_chart.coordinates.index_select(0, indices))
            else:
                self.target.copy_(self._sample_bounds(self.output_chart.coordinates))
            self.amplitude.normal_(mean=0.0, std=1.0 / math.sqrt(self.atoms))

    def _sample_bounds(self, coordinates: Tensor) -> Tensor:
        low = coordinates.amin(dim=0)
        high = coordinates.amax(dim=0)
        unit = torch.rand(
            self.atoms,
            coordinates.shape[1],
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        return low + unit * (high - low)

    def _kernel_factors(self) -> tuple[Tensor, Tensor]:
        phi_input = self.input_kernel(self.input_chart.coordinates, self.source)
        phi_output = self.output_kernel(self.output_chart.coordinates, self.target)
        return phi_input, phi_output

    def dense_weight(self) -> Tensor:
        """Materialize the represented ``[out_features, in_features]`` matrix."""

        phi_input, phi_output = self._kernel_factors()
        return (phi_output * self.amplitude) @ phi_input.transpose(-2, -1)

    def _resolved_backend(self) -> Literal["factored", "materialized"]:
        if self.backend != "auto":
            return self.backend
        factor_size = self.atoms * (self.in_features + self.out_features)
        dense_size = self.in_features * self.out_features
        return "factored" if factor_size <= dense_size else "materialized"

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input shape [..., {self.in_features}], "
                f"got {tuple(inputs.shape)}"
            )

        if self._resolved_backend() == "materialized":
            return F.linear(inputs, self.dense_weight())

        phi_input, phi_output = self._kernel_factors()
        atom_values = (inputs @ phi_input) * self.amplitude
        return atom_values @ phi_output.transpose(-2, -1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"atoms={self.atoms}, atom_init={self.atom_init!r}, "
            f"backend={self.backend!r}"
        )
