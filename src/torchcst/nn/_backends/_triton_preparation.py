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


def route_and_layout(
    routing,
    decoded,
    *,
    support=None,
    retain_owners=True,
    batched_support=True,
    support_boxes=None,
    support_witness_cols=None,
    support_fast_witness=False,
):
    if support_witness_cols is not None and support_boxes is None:
        raise ValueError("support_witness_cols requires support_boxes")
    if support_fast_witness and (
        support_witness_cols is None or not routing.local_candidates
    ):
        raise ValueError("fast witness requires hints and local owner routing")
    count, stations = decoded.shape[0], routing.starts.numel()
    buckets = stations if support is None else 2 * stations + 1
    # Only the permutation and offsets survive preparation. Reuse ownership
    # storage for support keys when the caller does not need owners returned.
    key_dtype = torch.int32 if not retain_owners and buckets < 2**31 else torch.long
    owners = torch.empty(count, device=decoded.device, dtype=key_dtype)
    owner_rows = torch.empty_like(owners) if support_fast_witness else owners
    offsets = torch.empty(buckets + 1, device=decoded.device, dtype=torch.long)
    if not count:
        offsets.zero_()
        order = torch.empty(0, device=decoded.device, dtype=torch.long)
        return owners if retain_owners else None, order, offsets
    bg = (
        min(256, tr.next_power_of_2(stations))
        if stations > 1024
        else tr.next_power_of_2(stations)
    )
    ba = min(32, max(1, 1024 // bg))
    owner_kernel = kernels.owners_chunked if stations > 1024 else kernels.owners
    with torch.cuda.device(decoded.device):
        if routing.local_candidates:
            kernels.owners_local[(tr.cdiv(count, 128),)](
                decoded,
                routing.major_radius,
                routing.period,
                routing.starts,
                routing.spans,
                routing.spacing,
                routing.last_row,
                routing.pitch,
                owners,
                owner_rows,
                count,
                stations,
                *decoded.stride(),
                128,
                support[3] if support is not None else 0,
                support_fast_witness,
                enable_fp_fusion=False,
            )
        else:
            owner_kernel[(tr.cdiv(count, ba),)](
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
            keys = torch.empty_like(owners) if retain_owners else owners
            if support_boxes is not None:
                if not batched_support:
                    raise ValueError("boxed support requires batched support")
                support_kernel = kernels.support_buckets_batched_box
            else:
                support_kernel = (
                    kernels.support_buckets_batched
                    if batched_support
                    else kernels.support_buckets
                )
            batch_atoms = 8 if batched_support else 1
            support_kernel[(tr.cdiv(count, batch_atoms),)](
                decoded,
                precision,
                owners,
                circle,
                section,
                *((support_boxes,) if support_boxes is not None else ()),
                *((owner_rows,) if support_boxes is not None else ()),
                *(
                    (
                        support_witness_cols
                        if support_witness_cols is not None
                        else support_boxes,
                    )
                    if support_boxes is not None
                    else ()
                ),
                keys,
                circle.shape[0],
                section.shape[0],
                decoded.shape[1],
                stations,
                station_rows,
                *decoded.stride(),
                tr.next_power_of_2(station_rows),
                min(256, tr.next_power_of_2(section.shape[0]))
                if batched_support
                else 256,
                **({"A": count, "BA": batch_atoms} if batched_support else {}),
                **(
                    {
                        "USE_WITNESS": support_witness_cols is not None,
                        "FAST_WITNESS": support_fast_witness,
                    }
                    if support_boxes is not None
                    else {}
                ),
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
    return owners if retain_owners else None, order, offsets


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
