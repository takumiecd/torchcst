"""A fixed-shape sum of kernel-defined operator atoms."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchcst._derivatives import AtomDerivatives, AutogradFrameGeometry
from torchcst.atoms import Atoms
from torchcst.geometry import Chart, StripChart
from torchcst.kernels import AtomInit, Kernel
from torchcst.profiling import cst_span

from .atom_grad import LinearAtomGrad
from .module import CSTModule, RepulsionKind

Backend = Literal["auto", "factored", "materialized"]


class CSTLinear(CSTModule):
    r"""Represent a linear map as ``sum(kernel(p[a]))``."""

    def __init__(
        self,
        input_chart: Chart | None = None,
        output_chart: Chart | None = None,
        *,
        chart: Chart | None = None,
        atoms: int,
        kernel: Kernel,
        atom_init: AtomInit = "balanced",
        backend: Backend = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if chart is not None:
            if input_chart is not None or output_chart is not None:
                raise ValueError(
                    "chart cannot be combined with input_chart or output_chart"
                )
            input_chart = chart
        if not isinstance(input_chart, Chart) or (
            output_chart is not None and not isinstance(output_chart, Chart)
        ):
            raise TypeError("charts must be Chart instances")
        single_chart = output_chart is None
        if single_chart and (
            not hasattr(input_chart, "shape") or len(input_chart.shape) != 2
        ):
            raise ValueError("a single chart must describe an [out, in] operator")
        if single_chart and not callable(getattr(kernel, "weight", None)):
            raise TypeError("a single-chart kernel must implement weight(chart, p)")
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
        if backend == "factored" and (
            single_chart or not kernel.supports_factorization
        ):
            raise ValueError(
                "the selected kernel does not support factorized execution"
            )

        if single_chart:
            self.chart = input_chart
        else:
            self.input_chart = input_chart
            self.output_chart = output_chart
        self.kernel = kernel
        self.atom_init = atom_init
        self.backend = backend

        reference = input_chart.reference if single_chart else input_chart.coordinates
        target_device = device or reference.device
        target_dtype = dtype or reference.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("CSTLinear requires a floating-point dtype")
        self.to(device=target_device, dtype=target_dtype)

        charts = self.cst_charts()
        parameter_dim = kernel.parameter_dim(*charts)
        if parameter_dim < 1:
            raise ValueError("kernel.parameter_dim must be positive")
        p = kernel.initialize(*charts, atoms, mode=atom_init)
        expected_shape = (atoms, parameter_dim)
        if p.shape != expected_shape:
            raise ValueError(
                f"kernel.initialize must return shape {list(expected_shape)}"
            )
        if p.device != torch.device(target_device) or p.dtype != target_dtype:
            raise ValueError("kernel.initialize must match the module device and dtype")
        self.atoms = Atoms(p)

    def _checkpoint_layout(self) -> dict[str, object]:
        layout = {"in_features": self.in_features, "out_features": self.out_features}
        if hasattr(self, "chart"):
            layout["chart_mode"] = "single"
        return layout

    @property
    def in_features(self) -> int:
        return (
            self.chart.shape[1] if hasattr(self, "chart") else self.input_chart.features
        )

    @property
    def out_features(self) -> int:
        return (
            self.chart.shape[0]
            if hasattr(self, "chart")
            else self.output_chart.features
        )

    @property
    def atom_count(self) -> int:
        return self.atoms.count

    def materialized_atoms(self) -> Tensor:
        """Return the complete operator contribution of each atom."""

        return self._materialize_atoms(self.atoms.p)

    def _materialize_atoms(self, p: Tensor) -> Tensor:
        represented = self.kernel.materialize_atoms(*self.cst_charts(), p)
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

        if any(chart.trainable for chart in self.cst_charts()):
            raise ValueError("the first derivative engine supports frozen charts only")
        return AtomDerivatives(
            self.atoms,
            self._materialize_atoms,
            factor_atoms=(
                self._factor_atoms
                if len(self.cst_charts()) == 2 and self.kernel.supports_factorization
                else None
            ),
        )

    def _factor_atoms(self, p: Tensor) -> tuple[Tensor, Tensor]:
        if hasattr(self, "chart"):
            raise NotImplementedError(
                "single-chart kernels have no input/output factors"
            )
        return self.kernel.factors(self.input_chart, self.output_chart, p)

    def cst_frame_geometry(self) -> AutogradFrameGeometry:
        """Build the optimizer-independent representation-frame geometry."""

        return AutogradFrameGeometry(self.cst_derivatives())

    def cst_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the fixed-shape parameters owned by this CST site."""

        return (self.atoms.p,)

    def cst_charts(self) -> tuple[Chart, ...]:
        """Return the chart or legacy chart pair observed by this site."""

        return (
            (self.chart,)
            if hasattr(self, "chart")
            else (self.input_chart, self.output_chart)
        )

    def dense_weight(self) -> Tensor:
        """Materialize the canonical sum of complete kernel atoms."""

        if hasattr(self, "chart"):
            return self.kernel.weight(self.chart, self.atoms.p)
        return self.materialized_atoms().sum(dim=0)

    def packed_weight(self) -> Tensor:
        """Return tile-major storage for a single StripChart operator."""

        if not hasattr(self, "chart") or not isinstance(self.chart, StripChart):
            raise TypeError("packed_weight requires a single StripChart")
        if not callable(getattr(self.kernel, "packed_weight", None)):
            raise TypeError("kernel does not provide tile-major materialization")
        return self.kernel.packed_weight(self.chart, self.atoms.p)

    def repulsion_terms(
        self, *, kind: RepulsionKind = "cosine"
    ) -> tuple[Tensor, Tensor]:
        """Return ``(S, κ)`` for the atom-operator repulsion identity.

        ``S`` has the realized operator shape ``[out, in]``. ``κ`` is a scalar.
        Both are accumulated from the same materialized atoms, so the energy
        ``||S||_F^2 - κ`` equals the off-diagonal pair sum without an
        ``O(K^2)`` Gram matrix. ``kind="raw"`` uses the realized atoms;
        ``kind="cosine"`` uses Frobenius-normalized atoms; ``kind="abs"``
        uses the elementwise absolute atoms. Zero-norm atoms are omitted
        from the cosine sum. For ``abs``, ``κ`` matches raw.
        """

        if kind not in ("cosine", "raw", "abs"):
            raise ValueError("kind must be 'cosine', 'raw', or 'abs'")
        atoms = self.materialized_atoms()
        if kind == "cosine":
            norms = torch.linalg.vector_norm(atoms, dim=(1, 2), keepdim=True)
            scale = torch.where(norms > 0, norms.reciprocal(), torch.zeros_like(norms))
            atoms = atoms * scale
        elif kind == "abs":
            atoms = atoms.abs()
        summed = atoms.sum(dim=0)
        kappa = atoms.square().sum()
        return summed, kappa

    def _resolved_backend(self) -> Literal["factored", "materialized"]:
        if hasattr(self, "chart"):
            return "materialized"
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
        if hasattr(self, "chart"):
            with cst_span("cst.linear.materialize"):
                weight = self.kernel.weight(self.chart, p)
            with cst_span("cst.linear.dense_matmul"):
                return F.linear(inputs, weight)
        if backend == "materialized":
            with cst_span("cst.linear.materialize"):
                weight = self._materialize_atoms(p).sum(dim=0)
            with cst_span("cst.linear.dense_matmul"):
                return F.linear(inputs, weight)

        with cst_span("cst.linear.factors"):
            phi_input, phi_output = self.kernel.factors(
                self.input_chart, self.output_chart, p
            )
        with cst_span("cst.linear.input_matmul"):
            atom_values = inputs @ phi_input
        with cst_span("cst.linear.output_matmul"):
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
