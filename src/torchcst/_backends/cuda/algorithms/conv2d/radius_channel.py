"""Experimental weight-only CUDA path for capped M8 channel charts."""

import torch
from torch.autograd.function import once_differentiable

from ..linear.radius_cell_mass.executor import _extension


class _ChannelWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, physical, input_sites, first_fourier_level):
        native = _extension(8, rational_wide=True)
        weight, geometry = native.packed_channel_weight_forward(
            physical, input_sites, first_fourier_level
        )
        ctx.save_for_backward(physical, geometry)
        ctx.input_sites = input_sites
        ctx.first_fourier_level = first_fourier_level
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        physical, geometry = ctx.saved_tensors
        native = _extension(8, rational_wide=True)
        grad = native.packed_channel_weight_grad(
            physical,
            cotangent.contiguous(),
            geometry,
            ctx.input_sites,
            ctx.first_fourier_level,
        )
        return grad, None, None


def channel_weights(physical, input_sites, *, first_fourier_level):
    """Return [tap, 128, input_sites] from [tap, atoms, a/R/co/ci].

    Radii are in (0, 32]. Channel charts use unit spacing, base tile one,
    and merge factor two. First Fourier level 6 selects direct accumulation
    on the entire declared hierarchy. The caller owns the rational decoder.
    """
    if type(input_sites) is not int or input_sites not in (96, 128):
        raise ValueError("input_sites must be 96 or 128")
    if type(first_fourier_level) is not int or not 3 <= first_fourier_level <= 6:
        raise ValueError("first_fourier_level must be in [3, 6]")
    if (
        physical.device.type != "cuda"
        or physical.dtype != torch.float32
        or physical.ndim != 3
        or physical.shape[0] not in (1, 9)
        or not 0 < physical.shape[1] <= 4096
        or physical.shape[2] != 4
    ):
        raise ValueError("expected CUDA FP32 physical atoms [1 or 9, 1..4096, 4]")
    return _ChannelWeight.apply(physical.contiguous(), input_sites, first_fourier_level)
