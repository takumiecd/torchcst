"""A fixed-shape sum of kernel-defined operator atoms."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchcst._derivatives import AtomDerivatives
from torchcst.atoms import Atoms
from torchcst.geometry import Chart
from torchcst.kernels import AtomInit, Kernel

from .atom_grad import LinearAtomGrad

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

    def _forward_from_p(
        self,
        inputs: Tensor,
        p: Tensor,
        *,
        backend: Literal["factored", "materialized"],
    ) -> Tensor:
        if backend == "materialized":
            return F.linear(inputs, self._materialize_atoms(p).sum(dim=0))

        phi_input, phi_output = self.kernel.factors(
            self.input_chart, self.output_chart, p
        )
        atom_values = inputs @ phi_input
        return atom_values @ phi_output.transpose(-2, -1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input shape [..., {self.in_features}], "
                f"got {tuple(inputs.shape)}"
            )

        atom_grad = self.atoms.grad
        if atom_grad is not None and atom_grad.active:
            if not isinstance(atom_grad, LinearAtomGrad):
                raise TypeError("CSTLinear requires an active LinearAtomGrad")
            outputs = atom_grad.apply(self, inputs)
        else:
            outputs = self._forward_from_p(
                inputs,
                self.atoms.p,
                backend=self._resolved_backend(),
            )
        return outputs

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"atoms={self.atom_count}, parameter_dim={self.atoms.parameter_dim}, "
            f"atom_init={self.atom_init!r}, backend={self.backend!r}"
        )
