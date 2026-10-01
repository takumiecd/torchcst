"""Collect tensor metadata; no parameter-value reads or implicit device-zero selection."""

from functools import cache

import torch

from .schema import DeviceInfo, DispatchContext, PrecisionPolicy, RequiredGrads


@cache
def _cuda_device(index):
    prop = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        "cuda", index, prop.name, (prop.major, prop.minor), prop.multi_processor_count
    )


def context_from_tensors(operator, x, parameters, *, workspace_limit_bytes=None):
    if x.device != parameters.device or x.dtype != parameters.dtype:
        raise ValueError("input and parameters must have the same device and dtype")
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("parameters must have shape [atoms, 5]")
    cuda = x.device.type == "cuda"
    device = (
        _cuda_device(x.device.index)
        if cuda
        else DeviceInfo(x.device.type, x.device.index)
    )
    return DispatchContext(
        operator=operator,
        input_shape=tuple(x.shape),
        input_strides=tuple(x.stride()),
        dtype=x.dtype,
        atom_count=len(parameters),
        device=device,
        required_grads=RequiredGrads(
            torch.is_grad_enabled() and x.requires_grad,
            torch.is_grad_enabled() and parameters.requires_grad,
        ),
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
