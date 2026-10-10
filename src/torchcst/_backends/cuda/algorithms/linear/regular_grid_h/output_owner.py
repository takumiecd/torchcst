"""Forward-only spatial routing; one CTA owns each batch/output tile.

Sort coarse centre bins, not per-site support IDs. Rebuild from the prepared
forward snapshot every call. Routing tensors are temporary and never retained
for backward, which still uses the original atom-owned contractions.
"""

import torch


def forward_output_owned(x, packed, sizes, recipe):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    if not a:
        return x.new_zeros((b, no))
    y = x.new_empty((b, no))
    bins = tr.cdiv(no, recipe.output_tile)
    keys = torch.empty(a, device=x.device, dtype=torch.int32)
    distances = torch.empty(tr.cdiv(a, 256), device=x.device, dtype=torch.int32)
    kernels.routing_keys[(tr.cdiv(a, 256),)](
        packed,
        keys,
        distances,
        a,
        no,
        lo,
        oo,
        recipe.output_tile,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    sorted_keys, order = torch.sort(keys)
    boundaries = torch.searchsorted(
        sorted_keys, torch.arange(bins + 1, device=x.device, dtype=torch.int32)
    )
    max_distance = distances.amax()
    kernels.output_owned[(bins, tr.cdiv(b, recipe.batch_tile))](
        x,
        packed,
        order,
        boundaries,
        max_distance,
        y,
        a,
        b,
        ni,
        no,
        li,
        lo,
        oi,
        oo,
        *x.stride(),
        recipe.batch_tile,
        recipe.patch_sites,
        recipe.atom_group,
        recipe.output_tile,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y
