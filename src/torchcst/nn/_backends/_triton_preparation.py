"""GPU preparation preserving the Torch geometry and amplitude derivatives."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import _triton_preparation_kernels as kernels


def tile_parameters(kernel, p):
    # Keep clamp in Torch: its derivative includes the endpoints. Width is
    # intentionally detached, matching DirectAmpWidth._tile_parameters.
    maximum = kernel.amplitude_max
    amplitude = p[:, 0].clamp(-maximum, maximum)
    precision = torch.empty_like(amplitude)
    if p.shape[0]:
        with torch.cuda.device(p.device):
            kernels.bandwidth[(tr.cdiv(p.shape[0], 128),)](
                amplitude,
                p,
                kernel.sigma_min_input,
                kernel.sigma_birth_input,
                kernel.sigma_max_input,
                kernel.upper_floor_input,
                kernel.w_c,
                kernel.kappa,
                kernel.lower_kappa,
                kernel.upper_decay_power,
                precision,
                p.shape[0],
                *p.stride(),
                128,
                enable_fp_fusion=False,
            )
    return p[:, 2:], amplitude, precision


def route_and_layout(routing, decoded, *, support=None):
    count, stations = decoded.shape[0], routing.starts.numel()
    owners = torch.empty(count, device=decoded.device, dtype=torch.long)
    buckets = stations if support is None else 2 * stations + 1
    offsets = torch.empty(buckets + 1, device=decoded.device, dtype=torch.long)
    if not count:
        offsets.zero_()
        return owners, owners, offsets
    bg = tr.next_power_of_2(stations)
    ba = min(32, max(1, 1024 // bg))
    with torch.cuda.device(decoded.device):
        kernels.owners[(tr.cdiv(count, ba),)](
            decoded,
            routing.major_radius,
            routing.period,
            routing.starts,
            routing.spans,
            routing.spacing,
            routing.last_row,
            owners,
            count,
            stations,
            *decoded.stride(),
            ba,
            bg,
            enable_fp_fusion=False,
        )
        keys = owners
        if support is not None:
            circle, section, precision, station_rows = support
            keys = torch.empty_like(owners)
            kernels.support_buckets[(count,)](
                decoded,
                precision,
                owners,
                circle,
                section,
                keys,
                circle.shape[0],
                section.shape[0],
                decoded.shape[1],
                stations,
                station_rows,
                *decoded.stride(),
                tr.next_power_of_2(station_rows),
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        sorted_owners, order = torch.sort(keys, stable=True)
        kernels.offsets[(tr.cdiv(buckets + 1, 128),)](
            sorted_owners,
            offsets,
            count,
            buckets,
            count.bit_length(),
            128,
        )
    return owners, order, offsets


class Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, amplitude, precision, decoded, order):
        count, dimension = decoded.shape
        packed = decoded.new_empty((count, dimension + 2))
        ctx.save_for_backward(order)
        ctx.dimension = dimension
        if count:
            with torch.cuda.device(decoded.device):
                kernels.pack[(tr.cdiv(packed.numel(), 256),)](
                    amplitude,
                    precision,
                    decoded,
                    order,
                    packed,
                    count,
                    dimension,
                    amplitude.stride(0),
                    precision.stride(0),
                    *decoded.stride(),
                    256,
                )
        return packed

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (order,) = ctx.saved_tensors
        count, dimension = order.numel(), ctx.dimension
        da = gradient.new_empty(count)
        dc = gradient.new_empty((count, dimension))
        if count:
            with torch.cuda.device(gradient.device):
                kernels.unpack_grad[(tr.cdiv(count * (dimension + 1), 256),)](
                    gradient,
                    order,
                    da,
                    dc,
                    count,
                    dimension,
                    *gradient.stride(),
                    256,
                )
        return da, None, dc, None
