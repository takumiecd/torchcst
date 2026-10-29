"""Atom coordinate update inputs and a live-state binding, without Linear invocation data."""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from torchcst._backends.schema import DeviceInfo
from torchcst.geometry.spec import GeometrySpec
from torchcst.kernels.spec import KernelSpec

from .binding import Operator
from .metadata import operator_declaration


@dataclass(frozen=True)
class AtomUpdateInputs:
    """Old point and outer step; the binding holds the current optimizer proposal."""

    previous: Tensor
    step_size: float
    amplitude_weight_decay: float = 0.0


@dataclass(frozen=True)
class AtomUpdateContext:
    kernel: KernelSpec
    geometries: tuple[GeometrySpec, ...]
    parameter_shape: tuple[int, int]
    dtype: torch.dtype
    device: DeviceInfo
    contiguous: bool
    scalar_metadata: tuple[tuple[str, torch.dtype, DeviceInfo, int], ...]
    execution_mode: str = "eager"
    workspace_limit_bytes: int | None = None
    amplitude_decay: bool = False
    operation_id: str = field(default="atom_update", init=False)


def amplitude_decay_amount(context, inputs):
    """Validate the declared decay before an Algorithm writes any coordinates."""
    decay, rate = inputs.amplitude_weight_decay, inputs.step_size
    if (
        type(decay) not in (int, float)
        or not math.isfinite(decay)
        or decay < 0
        or type(rate) not in (int, float)
        or not math.isfinite(rate)
        or rate <= 0
        or rate * decay > 1
    ):
        raise ValueError("invalid amplitude_weight_decay or step_size")
    if context.amplitude_decay != (decay != 0):
        raise ValueError("amplitude decay inputs differ from the prepared context")
    if decay:
        from torchcst.kernels.parameterizations import (
            PolarAmpRadiusSpec,
            PolarAmpWidthSpec,
        )

        if type(context.kernel.parameterization) not in (
            PolarAmpWidthSpec,
            PolarAmpRadiusSpec,
        ):
            raise ValueError("amplitude decay requires Polar atoms")
    return rate * decay


class AtomUpdateBinding:
    """Reference an existing Operator's state; own only transient AlgorithmState.

    No Parameter, optimizer moments or old-point snapshot is copied into this
    binding. The caller owns each invocation's previous point. Updates write the
    bound Parameter in place; the base optimizer owns its proposal and clocks.
    """

    input_type = AtomUpdateInputs

    def __init__(self, operator: Operator, *, workspace_limit_bytes=None):
        if not isinstance(operator, Operator):
            raise TypeError("AtomUpdateBinding requires a live Operator")
        self.operator = operator
        self.workspace_limit_bytes = workspace_limit_bytes
        self._algorithm_states = []
        self.execution_declaration()

    @property
    def atom_state(self):
        return self.operator.atoms.__dict__.get("_atom_state_owner")

    @property
    def kernel(self):
        return self.operator.kernel

    def cst_charts(self):
        return self.operator.charts

    def declaration(self):
        return self.operator.declaration()

    def execution_parameters(self):
        return self.operator.p

    def execution_declaration(self):
        return operator_declaration(self)

    def validate_inputs(self, inputs):
        if type(inputs) is not AtomUpdateInputs or not isinstance(
            inputs.previous, Tensor
        ):
            raise TypeError("AtomUpdateInputs requires a Tensor previous point")
        previous, proposal = inputs.previous, self.execution_parameters()
        decay = inputs.amplitude_weight_decay
        if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
            raise ValueError("amplitude_weight_decay must be finite and nonnegative")
        if (
            type(inputs.step_size) not in (int, float)
            or not math.isfinite(inputs.step_size)
            or inputs.step_size <= 0
        ):
            raise ValueError("step_size must be finite and positive")
        if inputs.step_size * decay > 1:
            raise ValueError("step_size * amplitude_weight_decay must not exceed 1")
        if decay:
            from torchcst.kernels.parameterizations import (
                PolarAmpRadiusSpec,
                PolarAmpWidthSpec,
            )

            if type(self.kernel.spec.parameterization) not in (
                PolarAmpWidthSpec,
                PolarAmpRadiusSpec,
            ):
                raise ValueError("amplitude_weight_decay requires Polar atoms")
        if (
            previous.layout != torch.strided
            or proposal.layout != torch.strided
            or previous.ndim != 2
            or previous.shape != proposal.shape
            or not previous.is_floating_point()
            or previous.dtype != proposal.dtype
            or previous.device != proposal.device
        ):
            raise ValueError(
                "previous point must match the bound atom shape, device and dtype"
            )
        if torch._C._is_alias_of(previous, proposal):
            raise ValueError("previous point must not alias the optimizer proposal")
        if previous.requires_grad:
            raise ValueError("previous point must be a detached optimizer snapshot")

    def build_context(self, inputs):
        declaration = self.execution_declaration()
        p = self.execution_parameters()
        cuda = p.is_cuda
        geometries = tuple(chart.geometry.spec for chart in self.operator.charts)
        metadata = self._scalar_metadata(declaration)
        mode = (
            "cuda_graph"
            if cuda and torch.cuda.is_current_stream_capturing()
            else "eager"
        )
        contiguous = p.is_contiguous() and inputs.previous.is_contiguous()
        decay = inputs.amplitude_weight_decay != 0
        key = (
            tuple(id(geometry) for geometry in geometries),
            tuple(p.shape),
            p.dtype,
            p.device,
            contiguous,
            mode,
            self.workspace_limit_bytes,
            decay,
        )
        validated = self.__dict__.get("_execution_declaration")
        cacheable = (
            validated is not None
            and validated[0] is not None
            and validated[1] is declaration
            and type(decay) is bool
            and (
                self.workspace_limit_bytes is None
                or type(self.workspace_limit_bytes) is int
            )
        )
        cached = self.__dict__.get("_context_cache")
        if (
            cacheable
            and cached is not None
            and cached[0] is validated
            and cached[1] is metadata
            and cached[2] == key
        ):
            return cached[3]
        context = AtomUpdateContext(
            kernel=declaration.kernel,
            geometries=geometries,
            parameter_shape=tuple(p.shape),
            dtype=p.dtype,
            device=DeviceInfo(p.device.type, p.device.index),
            contiguous=contiguous,
            scalar_metadata=metadata,
            execution_mode=mode,
            workspace_limit_bytes=self.workspace_limit_bytes,
            amplitude_decay=decay,
        )
        if cacheable:
            self.__dict__["_context_cache"] = (validated, metadata, key, context)
        else:
            self.__dict__.pop("_context_cache", None)
        return context

    def _scalar_metadata(self, declaration):
        """Reuse placement metadata only after the live declaration guard ran.

        The validated declaration entry keeps its identity on a cache hit.
        Buffer names and sizes need an extra guard: changing a registration's
        name can leave the fixed-state signature unchanged. This cache contains
        metadata only; invocation tensors and old-point snapshots are not kept.
        """
        owners = (self.kernel, *(chart.geometry for chart in self.cst_charts()))
        topology = tuple(
            tuple(
                (name, id(value), value.numel())
                for name, value in owner._buffers.items()
                if value is not None
            )
            for owner in owners
        )
        validated = self.__dict__.get("_execution_declaration")
        cacheable = (
            validated is not None
            and validated[0] is not None
            and validated[1] is declaration
        )
        previous = self.__dict__.get("_scalar_metadata_cache")
        if (
            cacheable
            and previous is not None
            and previous[0] is validated
            and previous[1] == topology
        ):
            return previous[2]
        metadata = tuple(
            (
                name if axis == 0 else f"geometry.{axis - 1}.{name}",
                value.dtype,
                DeviceInfo(value.device.type, value.device.index),
                value.numel(),
            )
            for axis, owner in enumerate(owners)
            for name, value in sorted(owner._buffers.items())
            if value is not None
        )
        if cacheable:
            self.__dict__["_scalar_metadata_cache"] = (validated, topology, metadata)
        else:
            self.__dict__.pop("_scalar_metadata_cache", None)
        return metadata

    def prepared_dispatch_signature(self):
        from torchcst._backends.dispatch.prepared import placement_signature

        return placement_signature(self)

    def state_signature(self):
        from torchcst._backends.state import tensor_signature

        return tensor_signature(
            self.execution_parameters()
        ), self.execution_declaration()

    def algorithm_state(self, algorithm, *, recipe):
        for state in self._algorithm_states:
            if state.algorithm is algorithm and state.recipe == recipe:
                return state
        state = algorithm.create_state(self, recipe=recipe)
        self._algorithm_states.append(state)
        return state

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_algorithm_states"] = []
        state.pop("_execution_declaration", None)
        state.pop("_prepared_dispatch", None)
        state.pop("_prepared_dispatch_stats", None)
        state.pop("_scalar_metadata_cache", None)
        state.pop("_context_cache", None)
        return state
