"""A fixed-shape sum of kernel-defined operator atoms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst._backends.linear import (
    Backend,
    ResolvedBackend,
    linear_forward,
    resolve_backend,
    validate_backend,
)
from torchcst._backends.cuda.dispatch.base import Selector
from torchcst._derivatives import AtomDerivatives, AutogradFrameGeometry
from torchcst.atoms import Atoms
from torchcst.charts import ChartSpec, ChartState, StripChartState, compile_chart
from torchcst.kernels import AtomInit, KernelSpec
from torchcst.kernels.options import KernelOptions
from torchcst.kernels.state import KernelState
from torchcst.operators import Operator

from .module import CSTModule, RepulsionKind


class CSTLinear(CSTModule):
    r"""Represent a linear map as ``sum(kernel(p[a]))``."""

    def __init__(
        self,
        input_chart: ChartSpec | ChartState | None = None,
        output_chart: ChartSpec | ChartState | None = None,
        *,
        chart: ChartSpec | ChartState | None = None,
        atoms: int | Tensor | Atoms,
        kernel: KernelSpec,
        kernel_options: KernelOptions | None = None,
        selector: Selector | None = None,
        atom_init: AtomInit = "balanced",
        backend: Backend = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        from torchcst._backends.torch.kernels import execution as _kernel

        super().__init__()
        if chart is not None:
            if input_chart is not None or output_chart is not None:
                raise ValueError(
                    "chart cannot be combined with input_chart or output_chart"
                )
            input_chart = chart
        if isinstance(input_chart, ChartSpec):
            input_chart = compile_chart(input_chart, device=device, dtype=dtype)
        if isinstance(output_chart, ChartSpec):
            output_chart = compile_chart(output_chart, device=device, dtype=dtype)
        if not isinstance(input_chart, ChartState) or (
            output_chart is not None and not isinstance(output_chart, ChartState)
        ):
            raise TypeError("charts must be ChartSpec or ChartState instances")
        single_chart = output_chart is None
        if single_chart and (
            not hasattr(input_chart, "shape") or len(input_chart.shape) != 2
        ):
            raise ValueError("a single chart must describe an [out, in] operator")
        if isinstance(atoms, bool) or not isinstance(atoms, (int, Tensor, Atoms)):
            raise TypeError("atoms must be an integer, Tensor, Parameter or Atoms")
        if isinstance(atoms, int) and atoms < 1:
            raise ValueError("atoms must be positive")
        if selector is not None and not isinstance(selector, Selector):
            raise TypeError("selector must implement Selector")
        self.selector = selector
        if not isinstance(kernel, KernelSpec):
            raise TypeError("kernel must be a KernelSpec")
        if atom_init not in ("balanced", "uniform"):
            raise ValueError("atom_init must be 'balanced' or 'uniform'")
        if single_chart:
            self.chart = input_chart
        else:
            self.input_chart = input_chart
            self.output_chart = output_chart
        self.kernel = KernelState(kernel, options=kernel_options)
        kernel = self.kernel
        self.atom_init = atom_init
        self.backend = backend

        reference = (
            atoms.p
            if isinstance(atoms, Atoms)
            else (atoms if isinstance(atoms, Tensor) else input_chart.reference)
        )
        target_device = device or reference.device
        target_dtype = dtype or reference.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("CSTLinear requires a floating-point dtype")
        self.to(device=target_device, dtype=target_dtype)

        charts = self.cst_charts()
        parameter_dim = _kernel.parameter_dim(kernel, *charts)
        if parameter_dim < 1:
            raise ValueError("kernel.parameter_dim must be positive")
        if isinstance(atoms, int):
            p = _kernel.initialize(kernel, *charts, atoms, mode=atom_init)
            atom_state = Atoms(p)
        elif isinstance(atoms, Atoms):
            atom_state = atoms
            p = atoms.p
        else:
            p = (
                atoms
                if isinstance(atoms, nn.Parameter)
                else atoms.to(device=charts[0].reference.device, dtype=target_dtype)
            )
            atom_state = Atoms(p)
        expected_shape = (atom_state.count, parameter_dim)
        if p.shape != expected_shape:
            raise ValueError(f"atom parameters must have shape {list(expected_shape)}")
        # A device alias such as "cuda" resolves to an indexed device after
        # .to(); compare against the actual chart device, not the alias.
        initialized_device = charts[0].reference.device
        if p.device != initialized_device or p.dtype != target_dtype:
            raise ValueError(
                "atom parameters must match the module device and dtype; convert shared parameters before binding"
            )
        self.atoms = atom_state
        # A plain view keeps existing Parameter/buffer ownership and state keys.
        self.__dict__["_operator"] = Operator(
            charts=self.cst_charts(), kernel=self.kernel, atoms=self.atoms
        )
        # Bind fixed metadata before Graph capture; parameters remain live.
        self._resolved_backend()

    @property
    def operator(self) -> Operator:
        """The live operator view; declaration snapshots remain explicit."""
        operator = self.__dict__["_operator"]
        charts = self.cst_charts()
        if (
            operator.kernel is not self.kernel
            or operator.atoms is not self.atoms
            or len(operator.charts) != len(charts)
            or any(a is not b for a, b in zip(operator.charts, charts))
        ):
            # Replacing configuration is an explicit boundary, not a per-step
            # scalar snapshot. Ordinary Tensor updates/conversions need no bind.
            operator = Operator(charts=charts, kernel=self.kernel, atoms=self.atoms)
            self.__dict__["_operator"] = operator
        return operator

    def declaration(self):
        """Snapshot the complete linear contract outside forward/Graph capture."""
        return self.operator.declaration()

    @property
    def backend(self) -> Backend:
        return self._backend

    @backend.setter
    def backend(self, value: Backend) -> None:
        validate_backend(value, self.cst_charts(), self.kernel)
        self._backend = value

    def _checkpoint_layout(self) -> dict[str, object]:
        layout = {
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
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
        return self.operator.materialize_atoms(p)

    def cst_derivatives(self) -> AtomDerivatives:
        """Build the internal atom-structured derivative operator."""
        from torchcst._backends.torch.kernels import execution as _kernel

        if any(chart.trainable for chart in self.cst_charts()):
            raise ValueError("the first derivative engine supports frozen charts only")
        return AtomDerivatives(
            self.atoms,
            self._materialize_atoms,
            factor_atoms=(
                self._factor_atoms
                if len(self.cst_charts()) == 2
                and _kernel.supports_factorization(self.kernel)
                else None
            ),
        )

    def _factor_atoms(self, p: Tensor) -> tuple[Tensor, Tensor]:
        if hasattr(self, "chart"):
            raise NotImplementedError(
                "single-chart kernels have no input/output factors"
            )
        return self.operator.factors(p)

    def cst_frame_geometry(self) -> AutogradFrameGeometry:
        """Build the optimizer-independent representation-frame geometry."""

        return AutogradFrameGeometry(self.cst_derivatives())

    def cst_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the fixed-shape parameters owned by this CST site."""

        return (self.atoms.p,)

    def cst_charts(self) -> tuple[ChartState, ...]:
        """Return the chart or chart pair observed by this site."""

        return (
            (self.chart,)
            if hasattr(self, "chart")
            else (self.input_chart, self.output_chart)
        )

    def dense_weight(self) -> Tensor:
        """Materialize the canonical sum of complete kernel atoms."""

        return self.operator.weight()

    def packed_weight(self) -> Tensor:
        """Return tile-major storage for a single StripChart operator."""
        from torchcst._backends.torch.kernels import execution as _kernel

        if not hasattr(self, "chart") or not (isinstance(self.chart, StripChartState)):
            raise TypeError("packed_weight requires a single StripChart")
        return _kernel.packed_weight(self.kernel, self.chart, self.atoms.p)

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

    def _resolved_backend(self) -> ResolvedBackend:
        return resolve_backend(self)

    def _forward_from_p(
        self, inputs: Tensor, p: Tensor, *, backend: ResolvedBackend
    ) -> Tensor:
        return linear_forward(self, inputs, p, backend=backend)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input shape [..., {self.in_features}], "
                f"got {tuple(inputs.shape)}"
            )

        return self._forward_from_p(
            inputs, self.atoms.p, backend=self._resolved_backend()
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"atoms={self.atom_count}, parameter_dim={self.atoms.parameter_dim}, "
            f"atom_init={self.atom_init!r}, backend={self.backend!r}"
        )
