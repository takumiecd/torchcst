"""Invocation metadata for CSTConv2d."""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from torchcst._backends.schema import DeviceInfo, PrecisionPolicy, RequiredGrads

from .context import _cuda_device
from .spec import OperatorSpec


@dataclass(frozen=True)
class Conv2dInputs:
    x: Tensor


@dataclass(frozen=True)
class Conv2dContext:
    operator: OperatorSpec
    atom_layout: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_shape: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: torch.dtype
    parameter_dtype: torch.dtype
    device: DeviceInfo
    atom_count: int
    parameter_dim: int
    required_grads: RequiredGrads
    precision: PrecisionPolicy
    execution_mode: str
    deterministic: bool
    workspace_limit_bytes: int | None = None
    operation_id: ClassVar[str] = "conv2d"

    @property
    def taps(self):
        return math.prod(self.kernel_size)

    @property
    def has_channel_taps(self):
        return self.atom_layout == "channel_taps" or self.kernel_size == (1, 1)


def conv2d_context(binding, inputs):
    x, p = inputs.x, binding.execution_parameters()
    cuda = x.is_cuda
    if cuda:
        with torch.cuda.device(x.device):
            device = _cuda_device(x.device.index)
            capturing = torch.cuda.is_current_stream_capturing()
    else:
        device = DeviceInfo(x.device.type, x.device.index)
        capturing = False
    grad = torch.is_grad_enabled()
    return Conv2dContext(
        operator=binding.execution_declaration(),
        atom_layout=binding.atom_layout,
        kernel_size=binding.kernel_size,
        stride=binding.stride,
        padding=binding.padding,
        dilation=binding.dilation,
        groups=binding.groups,
        input_shape=tuple(x.shape),
        input_strides=tuple(x.stride()),
        dtype=x.dtype,
        parameter_dtype=p.dtype,
        device=device,
        atom_count=p.shape[0],
        parameter_dim=p.shape[1],
        required_grads=RequiredGrads(
            grad and x.requires_grad, grad and p.requires_grad
        ),
        precision=PrecisionPolicy(
            autocast=cuda and torch.is_autocast_enabled(),
            allow_tf32=cuda
            and (
                torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            ),
        ),
        execution_mode="cuda_graph" if capturing else "eager",
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
