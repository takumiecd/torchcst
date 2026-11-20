"""CUDA pointwise application of two separable CST factors."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


def _rows(x: Tensor, stride: tuple[int, int]) -> Tensor:
    sampled = x[:, :, :: stride[0], :: stride[1]].permute(0, 2, 3, 1)
    return sampled.reshape(-1, x.shape[1]).contiguous()


def _image(rows: Tensor, shape: tuple[int, int, int], channels_last: bool) -> Tensor:
    image = rows.view(*shape, rows.shape[1]).permute(0, 3, 1, 2)
    return image if channels_last else image.contiguous()


def _project(rows: Tensor, left: Tensor, right: Tensor) -> Tensor:
    from .streamed_kernels import project

    return project(rows, left, right)


class _StreamedPointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phi_input, phi_output, stride):
        batch, _, height, width = x.shape
        sh, sw = stride
        spatial = ((height - 1) // sh + 1, (width - 1) // sw + 1)
        channels_last = x.is_contiguous(memory_format=torch.channels_last)
        rows = _rows(x, stride)
        need_x, need_input_factor, need_output_factor, _ = ctx.needs_input_grad
        ctx.save_for_backward(
            rows if need_input_factor or need_output_factor else None,
            phi_input if need_x or need_output_factor else None,
            phi_output if need_x or need_input_factor else None,
        )
        ctx.image_shape = (batch, *spatial)
        ctx.input_shape = x.shape
        ctx.stride = stride
        ctx.channels_last = channels_last
        output = _project(rows, phi_input, phi_output)
        return _image(output, ctx.image_shape, channels_last)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rows, phi_input, phi_output = ctx.saved_tensors
        need_x, need_input_factor, need_output_factor, _ = ctx.needs_input_grad
        grad_rows = _rows(grad_output, (1, 1))
        dx = None
        if need_x:
            dx_rows = _project(grad_rows, phi_output, phi_input)
            sampled = _image(dx_rows, ctx.image_shape, channels_last=True)
            stride_h, stride_w = ctx.stride
            if ctx.stride == (1, 1):
                dx = sampled if ctx.channels_last else sampled.contiguous()
            else:
                memory_format = (
                    torch.channels_last
                    if ctx.channels_last
                    else torch.contiguous_format
                )
                dx = torch.empty(
                    ctx.input_shape,
                    dtype=grad_rows.dtype,
                    device=grad_rows.device,
                    memory_format=memory_format,
                ).zero_()
                dx[:, :, ::stride_h, ::stride_w].copy_(sampled)

        # Contract a channel stripe of dY^T X with both factors before moving
        # to the next stripe. The same cotangent is used for both factor VJPs.
        di = do = None
        if need_input_factor or need_output_factor:
            out_channels = grad_rows.shape[1]
            if need_input_factor:
                di = rows.new_zeros((rows.shape[1], phi_output.shape[1]))
            if need_output_factor:
                do = rows.new_empty((out_channels, phi_input.shape[1]))
            stripe_size = min(32, (out_channels + 1) // 2)
            for start in range(0, out_channels, stripe_size):
                stripe = slice(start, start + stripe_size)
                cotangent = grad_rows[:, stripe].T @ rows
                if need_input_factor:
                    di.addmm_(cotangent.T, phi_output[stripe])
                if need_output_factor:
                    torch.mm(cotangent, phi_input, out=do[stripe])

        return dx, di, do, None


def streamed_pointwise(
    x: Tensor, phi_input: Tensor, phi_output: Tensor, stride: tuple[int, int]
) -> Tensor:
    """Sample an NCHW image and apply its factors without an assembled weight."""
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or phi_input.dtype != x.dtype
        or phi_output.dtype != x.dtype
        or torch.is_autocast_enabled("cuda")
    ):
        raise ValueError("streamed CSTConv2d requires CUDA FP32 with autocast disabled")
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise ValueError("streamed CSTConv2d requires NVIDIA compute capability 8.0+")
    if min(x.shape[2:]) < 1:
        raise ValueError("streamed CSTConv2d requires nonempty spatial dimensions")
    with torch.cuda.device(x.device):
        return _StreamedPointwise.apply(x, phi_input, phi_output, stride)
