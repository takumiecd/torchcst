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
        metadata = tuple(
            (
                name,
                value.dtype,
                DeviceInfo(value.device.type, value.device.index),
                value.numel(),
            )
            for name, value in sorted(self.kernel._buffers.items())
            if value is not None
        )
        # Geometry buffers can also be live kernel inputs. Include their
        # placement metadata so an incompatible Plan is rejected before Adam
        # mutates its proposal/state, without reading any scalar to the host.
        metadata += tuple(
            (
                f"geometry.{axis}.{name}",
                value.dtype,
                DeviceInfo(value.device.type, value.device.index),
                value.numel(),
            )
            for axis, chart in enumerate(self.cst_charts())
            for name, value in sorted(chart.geometry._buffers.items())
            if value is not None
        )
        return AtomUpdateContext(
            kernel=declaration.kernel,
            geometries=tuple(chart.geometry.spec for chart in self.operator.charts),
            parameter_shape=tuple(p.shape),
            dtype=p.dtype,
            device=DeviceInfo(p.device.type, p.device.index),
            contiguous=p.is_contiguous() and inputs.previous.is_contiguous(),
            scalar_metadata=metadata,
            execution_mode="cuda_graph"
            if cuda and torch.cuda.is_current_stream_capturing()
            else "eager",
            workspace_limit_bytes=self.workspace_limit_bytes,
            amplitude_decay=inputs.amplitude_weight_decay != 0,
        )

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
        return state
