"""Experimental weight-only CUDA path for capped M8 channel charts."""

import math

import torch
from torch.autograd.function import once_differentiable

from ..linear.radius_cell_mass.executor import _extension


class _ChannelRationalParameters(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, radius_min, radius_birth, radius_cap, amplitude_max):
        native = _extension(8, rational_wide=True)
        physical, jacobian = native.packed_channel_rational_decode(
            rows, radius_min, radius_birth, radius_cap, amplitude_max
        )
        ctx.save_for_backward(jacobian)
        return physical

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        (jacobian,) = ctx.saved_tensors
        native = _extension(8, rational_wide=True)
        grad = native.packed_channel_rational_grad(cotangent.contiguous(), jacobian)
        return grad, None, None, None, None


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
    def forward(
        ctx,
        physical,
        input_sites,
        output_sites,
        radius_cap,
        sort_owners,
        active_banks_only,
    ):
        native = _extension(8, rational_wide=True)
        weight, geometry, order, mask = native.packed_channel_fourier_forward(
            physical,
            input_sites,
            output_sites,
            radius_cap,
            sort_owners,
            active_banks_only,
        )
        ctx.save_for_backward(physical, geometry, order)
        ctx.chart = input_sites, output_sites, radius_cap
        ctx.mask = mask
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        physical, geometry, order = ctx.saved_tensors
        native = _extension(8, rational_wide=True)
        grad = native.packed_channel_fourier_grad(
            physical, cotangent.contiguous(), geometry, order, *ctx.chart, ctx.mask
        )
        return grad, None, None, None, None, None


def _validate_atoms(physical):
    if (
        physical.device.type != "cuda"
        or physical.dtype != torch.float32
        or physical.ndim != 3
        or physical.shape[0] not in (1, 9)
        or not 0 < physical.shape[1] <= 4096
        or physical.shape[2] != 4
    ):
        raise ValueError("expected CUDA FP32 channel Atom rows [1 or 9, 1..4096, 4]")


def channel_rational_parameters(
    rows, *, radius_min, radius_birth, radius_cap, amplitude_max
):
    """Decode CUDA rational Atom rows without a host read of their values."""
    if (
        type(radius_cap) is not int
        or not 1 <= radius_cap <= 64
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (radius_min, radius_birth, amplitude_max)
        )
        or not 0.5 <= radius_min <= radius_birth <= radius_cap
        or not 0 < amplitude_max <= 32
    ):
        raise ValueError("invalid channel rational radius bounds")
    _validate_atoms(rows)
    return _ChannelRationalParameters.apply(
        rows.contiguous(), radius_min, radius_birth, radius_cap, amplitude_max
    )


def channel_weights(physical, input_sites, *, first_fourier_level):
    """Return [tap, 128, input_sites] from [tap, atoms, a/R/co/ci].

    Radii are in (0, 32]. Channel charts use unit spacing, base tile one,
    and merge factor two. First Fourier level 6 selects direct accumulation
    on the entire declared hierarchy. The caller owns the rational decoder.
    """
    if type(first_fourier_level) is not int or not 3 <= first_fourier_level <= 6:
        raise ValueError("first_fourier_level must be in [3, 6]")
    if type(input_sites) is not int or input_sites not in (96, 128):
        raise ValueError("input_sites must be 96 or 128")
    _validate_atoms(physical)
    return _ChannelWeight.apply(physical.contiguous(), input_sites, first_fourier_level)


def channel_fourier_weights(
    physical,
    input_sites,
    *,
    output_sites=128,
    radius_cap=None,
    sort_owners=False,
    active_banks_only=False,
):
    """Assemble [tap, output_sites, input_sites] using Fourier coefficients.

    The M8 channel tori use unit-spaced cells and dyadic tiles starting at one
    cell. Physical radii must be in (0, radius_cap]. The default cap is 32 or
    half the shorter channel period, whichever is smaller. An explicit cap
    must be a power of two dividing both periods and at most half each.
    Sorting changes only the execution order. Eager active-bank selection
    reads a bitmask before allocating scratch.
    """
    if type(sort_owners) is not bool:
        raise ValueError("sort_owners must be a bool")
    if type(active_banks_only) is not bool:
        raise ValueError("active_banks_only must be a bool")
    for name, sites in (("input_sites", input_sites), ("output_sites", output_sites)):
        if type(sites) is not int or sites not in (16, 32, 64, 96, 128):
            raise ValueError(f"{name} must be 16, 32, 64, 96 or 128")
    max_radius = min(input_sites, output_sites) // 2
    if radius_cap is None:
        radius_cap = min(32, max_radius)
    if (
        type(radius_cap) is not int
        or radius_cap not in (1, 2, 4, 8, 16, 32, 64)
        or radius_cap > max_radius
        or input_sites % radius_cap
        or output_sites % radius_cap
    ):
        raise ValueError(
            "radius_cap must be a dyadic tile dividing both channel tori and at most their half-period"
        )
    _validate_atoms(physical)
    return _ChannelFourierWeight.apply(
        physical.contiguous(),
        input_sites,
        output_sites,
        radius_cap,
        sort_owners,
        active_banks_only,
    )
