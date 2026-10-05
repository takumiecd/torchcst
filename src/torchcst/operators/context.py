"""Linear invocation metadata and validation, independent of Algorithm selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache

import torch

from torchcst._backends.schema import (
    DeviceInfo,
    PrecisionPolicy,
    RequiredGrads,
)
from torchcst.operators import spec as operators


@dataclass(frozen=True)
class LinearContext:
    operator: operators.OperatorSpec
    input_shape: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: torch.dtype
    atom_count: int
    device: DeviceInfo
    parameter_dim: int
    required_grads: RequiredGrads = field(default_factory=RequiredGrads)
    execution_mode: str = "eager"
    deterministic: bool = False
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    workspace_limit_bytes: int | None = None

    def __post_init__(self):
        if not isinstance(self.operator, operators.OperatorSpec):
            raise TypeError("operator must be a common OperatorSpec")
        if type(self.input_shape) is not tuple or type(self.input_strides) is not tuple:
            raise TypeError("shape and strides must be immutable tuples")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if (
            not isinstance(self.device, DeviceInfo)
            or not isinstance(self.required_grads, RequiredGrads)
            or not isinstance(self.precision, PrecisionPolicy)
        ):
            raise TypeError("context requires typed device/gradient/precision metadata")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be bool")
        if type(self.parameter_dim) is not int or self.parameter_dim <= 0:
            raise ValueError("parameter dimension must be positive")
        if not self.input_shape or self.input_shape[-1] != self.operator.in_features:
            raise ValueError("input feature dimension differs from operator")
        if any(type(n) is not int or n < 0 for n in self.input_shape):
            raise ValueError("input shape must contain nonnegative integers")
        if len(self.input_strides) != len(self.input_shape) or any(
            type(s) is not int or s < 0 for s in self.input_strides
        ):
            raise ValueError("input strides must match shape")
        if type(self.atom_count) is not int or self.atom_count < 0:
            raise ValueError("atom count must be nonnegative")
        if self.execution_mode not in ("eager", "cuda_graph"):
            raise ValueError("unknown execution mode")
        if self.workspace_limit_bytes is not None and (
            type(self.workspace_limit_bytes) is not int
            or self.workspace_limit_bytes < 0
        ):
            raise ValueError("workspace limit must be nonnegative or None")

    @property
    def operation_id(self):
        return self.operator.operation_id

    @property
    def m(self):
        return math.prod(self.input_shape[:-1])


@cache
def _cuda_device(index):
    prop = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        "cuda", index, prop.name, (prop.major, prop.minor), prop.multi_processor_count
    )


def context_from_metadata(
    operator,
    *,
    input_shape,
    input_strides,
    dtype,
    atom_count,
    parameter_dim,
    device,
    required_grads=None,
    workspace_limit_bytes=None,
):
    cuda = device.type == "cuda"
    info = _cuda_device(device.index) if cuda else DeviceInfo(device.type, device.index)
    return LinearContext(
        operator=operator,
        input_shape=input_shape,
        input_strides=input_strides,
        dtype=dtype,
        atom_count=atom_count,
        parameter_dim=parameter_dim,
        device=info,
        required_grads=RequiredGrads() if required_grads is None else required_grads,
        execution_mode="cuda_graph"
        if cuda and torch.cuda.is_current_stream_capturing()
        else "eager",
        deterministic=torch.are_deterministic_algorithms_enabled(),
        precision=PrecisionPolicy(
            autocast=torch.is_autocast_enabled(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
        if cuda
        else PrecisionPolicy(),
        workspace_limit_bytes=workspace_limit_bytes,
    )


def context_from_tensors(operator, x, parameters, *, workspace_limit_bytes=None):
    if x.device != parameters.device or x.dtype != parameters.dtype:
        raise ValueError("input and parameters must have the same device and dtype")
    if parameters.ndim != 2 or parameters.shape[1] <= 0:
        raise ValueError("parameters must have shape [atoms, D] with D > 0")
    return context_from_metadata(
        operator,
        input_shape=tuple(x.shape),
        input_strides=tuple(x.stride()),
        dtype=x.dtype,
        atom_count=len(parameters),
        parameter_dim=parameters.shape[1],
        device=x.device,
        required_grads=RequiredGrads(
            torch.is_grad_enabled() and x.requires_grad,
            torch.is_grad_enabled() and parameters.requires_grad,
        ),
        workspace_limit_bytes=workspace_limit_bytes,
    )
