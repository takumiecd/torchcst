"""A fixed-shape sum of kernel-defined operator atoms."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchcst._derivatives import AtomDerivatives, RepresentedGradientAccumulator
from torchcst.atoms import Atoms
from torchcst.geometry import Chart
from torchcst.kernels import AtomInit, Kernel

Backend = Literal["auto", "factored", "materialized"]


class CSTLinear(nn.Module):
    r"""Represent a linear map as ``sum(kernel(p[a]))``."""

    def __init__(
        self,
        input_chart: Chart,
        output_chart: Chart,
        *,
        atoms: int,
        kernel: Kernel,
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
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must implement the Kernel contract")
        if tuple(kernel.parameters()):
            raise ValueError("a Kernel cannot own trainable state; put it in atom p")
        if atom_init not in ("balanced", "uniform"):
            raise ValueError("atom_init must be 'balanced' or 'uniform'")
        if backend not in ("auto", "factored", "materialized"):
            raise ValueError("backend must be 'auto', 'factored', or 'materialized'")
        if backend == "factored" and not kernel.supports_factorization:
            raise ValueError(
                "the selected kernel does not support factorized execution"
            )

        self.input_chart = input_chart
        self.output_chart = output_chart
        self.kernel = kernel
        self.atom_init = atom_init
        self.backend = backend

        target_device = device or input_chart.coordinates.device
        target_dtype = dtype or input_chart.coordinates.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("CSTLinear requires a floating-point dtype")
        self.to(device=target_device, dtype=target_dtype)

        parameter_dim = kernel.parameter_dim(input_chart, output_chart)
        if parameter_dim < 1:
            raise ValueError("kernel.parameter_dim must be positive")
        p = kernel.initialize(
            input_chart,
            output_chart,
            atoms,
            mode=atom_init,
        )
        expected_shape = (atoms, parameter_dim)
        if p.shape != expected_shape:
            raise ValueError(
                f"kernel.initialize must return shape {list(expected_shape)}"
            )
        if p.device != input_chart.coordinates.device or p.dtype != target_dtype:
            raise ValueError("kernel.initialize must match the module device and dtype")
        self.atoms = Atoms(p)
        self._represented_gradient = RepresentedGradientAccumulator(
            (self.out_features, self.in_features)
        )

    @property
    def in_features(self) -> int:
        return self.input_chart.features

    @property
    def out_features(self) -> int:
        return self.output_chart.features

    @property
    def atom_count(self) -> int:
        return self.atoms.count

    def materialized_atoms(self) -> Tensor:
        """Return the complete operator contribution of each atom."""

        return self._materialize_atoms(self.atoms.p)

    def _materialize_atoms(self, p: Tensor) -> Tensor:
        represented = self.kernel.materialize_atoms(
            self.input_chart, self.output_chart, p
        )
        if p.ndim != 2 or p.shape[1] != self.atoms.parameter_dim:
            raise ValueError(f"p must have shape [atoms, {self.atoms.parameter_dim}]")
        expected_shape = (p.shape[0], self.out_features, self.in_features)
        if represented.shape != expected_shape:
            raise ValueError(
                f"kernel.materialize_atoms must return shape {list(expected_shape)}"
            )
        return represented

    def cst_derivatives(self) -> AtomDerivatives:
        """Build the internal atom-structured derivative operator."""

        if self.input_chart.trainable or self.output_chart.trainable:
            raise ValueError("the first derivative engine supports frozen charts only")
        return AtomDerivatives(self.atoms, self._materialize_atoms)

    def cst_parameters(self) -> tuple[nn.Parameter]:
        """Return the fixed-shape parameters owned by this CST site."""

        return (self.atoms.p,)

    def enable_represented_gradient_capture(self, *, clear: bool = True) -> None:
        """Observe represented cotangents produced by subsequent backward calls."""

        self._represented_gradient.enable(clear=clear)

    def disable_represented_gradient_capture(self) -> None:
        """Stop observing represented cotangents without clearing the last value."""

        self._represented_gradient.disable()

    def clear_represented_gradient(self) -> None:
        """Discard the transient represented cotangent."""

        self._represented_gradient.clear()

    def represented_gradient(self) -> Tensor:
        """Return a detached snapshot of the accumulated ``dL/dW``."""

        return self._represented_gradient.value()

    def _observe_represented_gradient(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        if not self._represented_gradient.enabled or not outputs.requires_grad:
            return outputs

        saved_inputs = inputs.detach().clone()
        capture_generation = self._represented_gradient.generation

        def collect(output_gradient: Tensor) -> None:
            flat_inputs = saved_inputs.reshape(-1, self.in_features)
            flat_gradient = output_gradient.detach().reshape(-1, self.out_features)
            self._represented_gradient.add(
                flat_gradient.transpose(0, 1) @ flat_inputs,
                generation=capture_generation,
            )

        outputs.register_hook(collect)
        return outputs

    def dense_weight(self) -> Tensor:
        """Materialize the canonical sum of complete kernel atoms."""

        return self.materialized_atoms().sum(dim=0)

    def _resolved_backend(self) -> Literal["factored", "materialized"]:
        if self.backend != "auto":
            return self.backend
        if not self.kernel.supports_factorization:
            return "materialized"
        factor_size = self.atom_count * (self.in_features + self.out_features)
        dense_size = self.in_features * self.out_features
        return "factored" if factor_size <= dense_size else "materialized"

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input shape [..., {self.in_features}], "
                f"got {tuple(inputs.shape)}"
            )

        if self._resolved_backend() == "materialized":
            outputs = F.linear(inputs, self.dense_weight())
            return self._observe_represented_gradient(inputs, outputs)

        phi_input, phi_output = self.kernel.factors(
            self.input_chart, self.output_chart, self.atoms.p
        )
        atom_values = inputs @ phi_input
        outputs = atom_values @ phi_output.transpose(-2, -1)
        return self._observe_represented_gradient(inputs, outputs)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"atoms={self.atom_count}, parameter_dim={self.atoms.parameter_dim}, "
            f"atom_init={self.atom_init!r}, backend={self.backend!r}"
        )
