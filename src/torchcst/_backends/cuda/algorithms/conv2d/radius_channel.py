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


class _ChannelFourierWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, physical, input_sites, sort_owners, active_banks_only):
        native = _extension(8, rational_wide=True)
        weight, geometry, order, mask = native.packed_channel_fourier_forward(
            physical, input_sites, sort_owners, active_banks_only
        )
        ctx.save_for_backward(physical, geometry, order)
        ctx.input_sites = input_sites
        ctx.mask = mask
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        physical, geometry, order = ctx.saved_tensors
        native = _extension(8, rational_wide=True)
        grad = native.packed_channel_fourier_grad(
            physical, cotangent.contiguous(), geometry, order, ctx.input_sites, ctx.mask
        )
        return grad, None, None, None


def _validate_channel_shape(physical, input_sites):
    if type(input_sites) is not int or input_sites not in (96, 128):
        raise ValueError("input_sites must be 96 or 128")
    if (
        physical.device.type != "cuda"
        or physical.dtype != torch.float32
        or physical.ndim != 3
        or physical.shape[0] not in (1, 9)
        or not 0 < physical.shape[1] <= 4096
        or physical.shape[2] != 4
    ):
        raise ValueError("expected CUDA FP32 physical atoms [1 or 9, 1..4096, 4]")


def channel_weights(physical, input_sites, *, first_fourier_level):
    """Return [tap, 128, input_sites] from [tap, atoms, a/R/co/ci].

    Radii are in (0, 32]. Channel charts use unit spacing, base tile one,
    and merge factor two. First Fourier level 6 selects direct accumulation
    on the entire declared hierarchy. The caller owns the rational decoder.
    """
    if type(first_fourier_level) is not int or not 3 <= first_fourier_level <= 6:
        raise ValueError("first_fourier_level must be in [3, 6]")
    _validate_channel_shape(physical, input_sites)
    return _ChannelWeight.apply(physical.contiguous(), input_sites, first_fourier_level)


def channel_fourier_weights(
    physical, input_sites, *, sort_owners=False, active_banks_only=False
):
    """Assemble weights in Fourier coefficients at every level, across all taps.

    The Atom rows, radius cap, and charts match channel_weights. Optionally
    sort the execution indices by tap, bank, and periodic input/output owner
    tile. The Atom Parameter layout and gradient order follow the caller.
    Eager active-bank selection reads a bitmask before allocating scratch.
    """
    if type(sort_owners) is not bool:
        raise ValueError("sort_owners must be a bool")
    if type(active_banks_only) is not bool:
        raise ValueError("active_banks_only must be a bool")
    _validate_channel_shape(physical, input_sites)
    return _ChannelFourierWeight.apply(
        physical.contiguous(), input_sites, sort_owners, active_banks_only
    )
