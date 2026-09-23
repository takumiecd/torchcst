"""Two-dimensional convolution from a flattened local CST operator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchcst._derivatives import AtomDerivatives, AutogradFrameGeometry
from torchcst.atoms import Atoms
from torchcst.geometry import Chart
from torchcst.kernels import AtomInit, Kernel

from .module import CSTModule, RepulsionKind

Backend = Literal["auto", "factored", "materialized"]


def _pair(
    value: int | Sequence[int], *, name: str, allow_zero: bool = False
) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must have exactly two entries")
        values = tuple(value)
    else:
        values = (value, value)
    minimum = 0 if allow_zero else 1
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name}[{index}] must be an integer")
        if item < minimum:
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name}[{index}] must be {qualifier}")
    return values


class CSTConv2d(CSTModule):
    r"""Apply one flattened CST patch operator as a 2-D convolution.

    The kernel-defined operator has shape ``[out_channels, patch_features]``,
    where ``patch_features = (in_channels / groups) * kernel_height *
    kernel_width``. The module reshapes that local matrix to the conventional
    Conv2d weight only at the execution boundary.
    """

    def __init__(
        self,
        input_chart: Chart,
        output_chart: Chart,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        atoms: int,
        kernel: Kernel,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        atom_init: AtomInit = "balanced",
        backend: Backend = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(input_chart, Chart) or not isinstance(output_chart, Chart):
            raise TypeError("input_chart and output_chart must be Chart instances")
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("groups", groups),
            ("atoms", atoms),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
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
        if backend == "factored" and groups != 1:
            raise ValueError("factorized CSTConv2d currently requires groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size, name="kernel_size")
        self.stride = _pair(stride, name="stride")
        self.padding = _pair(padding, name="padding", allow_zero=True)
        self.dilation = _pair(dilation, name="dilation")
        self.groups = groups

        patch_features = (in_channels // groups) * math.prod(self.kernel_size)
        if input_chart.features != patch_features:
            raise ValueError(
                "input_chart must describe the flattened local patch: "
                f"expected {patch_features} features, got {input_chart.features}"
            )
        if output_chart.features != out_channels:
            raise ValueError(
                f"output_chart must have {out_channels} features, "
                f"got {output_chart.features}"
            )

        self.input_chart = input_chart
        self.output_chart = output_chart
        self.kernel = kernel
        self.atom_init = atom_init
        self.backend = backend

        target_device = device or input_chart.coordinates.device
        target_dtype = dtype or input_chart.coordinates.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("CSTConv2d requires a floating-point dtype")
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

    def _checkpoint_layout(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }

    @property
    def patch_features(self) -> int:
        return self.input_chart.features

    @property
    def atom_count(self) -> int:
        return self.atoms.count

    def _materialize_flat_atoms(self, p: Tensor) -> Tensor:
        represented = self.kernel.materialize_atoms(
            self.input_chart, self.output_chart, p
        )
        if p.ndim != 2 or p.shape[1] != self.atoms.parameter_dim:
            raise ValueError(f"p must have shape [atoms, {self.atoms.parameter_dim}]")
        expected_shape = (p.shape[0], self.out_channels, self.patch_features)
        if represented.shape != expected_shape:
            raise ValueError(
                f"kernel.materialize_atoms must return shape {list(expected_shape)}"
            )
        return represented

    def materialized_atoms(self) -> Tensor:
        """Return complete atoms as Conv2d weights ``[K, O, I/G, Kh, Kw]``."""

        return self._materialize_atoms(self.atoms.p)

    def _materialize_atoms(self, p: Tensor) -> Tensor:
        return self._materialize_flat_atoms(p).reshape(
            p.shape[0],
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )

    def dense_weight(self) -> Tensor:
        """Materialize the canonical sum as a Conv2d weight."""

        return self.materialized_atoms().sum(dim=0)

    def _factor_atoms(self, p: Tensor) -> tuple[Tensor, Tensor]:
        return self.kernel.factors(self.input_chart, self.output_chart, p)

    def cst_derivatives(self) -> AtomDerivatives:
        """Build derivatives of the flattened local CST operator."""

        if self.input_chart.trainable or self.output_chart.trainable:
            raise ValueError("the first derivative engine supports frozen charts only")
        return AtomDerivatives(
            self.atoms,
            self._materialize_flat_atoms,
            factor_atoms=self._factor_atoms
            if self.kernel.supports_factorization
            else None,
        )

    def cst_frame_geometry(self) -> AutogradFrameGeometry:
        """Build the optimizer-independent representation-frame geometry."""

        return AutogradFrameGeometry(self.cst_derivatives())

    def cst_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.atoms.p,)

    def cst_charts(self) -> tuple[Chart, ...]:
        return (self.input_chart, self.output_chart)

    def repulsion_terms(
        self, *, kind: RepulsionKind = "cosine"
    ) -> tuple[Tensor, Tensor]:
        if kind not in ("cosine", "raw", "abs"):
            raise ValueError("kind must be 'cosine', 'raw', or 'abs'")
        atoms = self.materialized_atoms()
        if kind == "cosine":
            dimensions = tuple(range(1, atoms.ndim))
            norms = torch.linalg.vector_norm(atoms, dim=dimensions, keepdim=True)
            scale = torch.where(
                norms > 0, norms.reciprocal(), torch.zeros_like(norms)
            )
            atoms = atoms * scale
        elif kind == "abs":
            atoms = atoms.abs()
        return atoms.sum(dim=0), atoms.square().sum()

    def _resolved_backend(self) -> Literal["factored", "materialized"]:
        if self.backend != "auto":
            return self.backend
        if not self.kernel.supports_factorization or self.groups != 1:
            return "materialized"
        factor_size = self.atom_count * (self.patch_features + self.out_channels)
        dense_size = self.patch_features * self.out_channels
        return "factored" if factor_size <= dense_size else "materialized"

    def _forward_from_p(
        self,
        inputs: Tensor,
        p: Tensor,
        *,
        backend: Literal["factored", "materialized"],
    ) -> Tensor:
        if backend == "materialized":
            weight = self._materialize_atoms(p).sum(dim=0)
            return F.conv2d(
                inputs,
                weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        phi_input, phi_output = self.kernel.factors(
            self.input_chart, self.output_chart, p
        )
        filters = phi_input.transpose(0, 1).reshape(
            p.shape[0], self.in_channels, *self.kernel_size
        )
        atom_values = F.conv2d(
            inputs,
            filters,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        mixing = phi_output.reshape(self.out_channels, p.shape[0], 1, 1)
        return F.conv2d(atom_values, mixing, bias=None)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected input shape [N, {self.in_channels}, H, W], "
                f"got {tuple(inputs.shape)}"
            )
        atom_grad = self.atoms.grad
        if atom_grad is not None and atom_grad.active:
            raise TypeError("CSTConv2d does not yet support an active AtomGrad program")
        return self._forward_from_p(
            inputs,
            self.atoms.p,
            backend=self._resolved_backend(),
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, atoms={self.atom_count}, "
            f"parameter_dim={self.atoms.parameter_dim}, "
            f"atom_init={self.atom_init!r}, backend={self.backend!r}"
        )
