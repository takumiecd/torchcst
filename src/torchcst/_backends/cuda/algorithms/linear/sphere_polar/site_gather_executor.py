"""Output-only CSR scratch; retain only the original ID/norm/geometry/H snapshots."""

import torch

from .direct_executor import _Direct
from .recompute_prepare import prepare_ids
from .site_gather_algorithm import INT32_MAX


def build_output_csr(index, count, n_sites, *, support_capacity=64):
    """Return device rowptr and atom IDs; only rowptr-delimited edges are live."""
    if type(support_capacity) is not int or support_capacity != 64:
        raise ValueError("requires CAP64")
    if type(n_sites) is not int or not 1 <= n_sites <= 32768:
        raise ValueError("requires 1..32768 output sites")
    atoms = len(count)
    capacity = atoms * support_capacity
    if capacity > INT32_MAX:
        raise ValueError("CSR edge capacity must fit signed int32")
    if (
        index.shape != (atoms, support_capacity)
        or count.shape != (atoms,)
        or index.dtype != torch.int16
        or count.dtype != torch.int32
        or index.device != count.device
        or not index.is_contiguous()
        or not count.is_contiguous()
    ):
        raise ValueError("requires contiguous int16 packed IDs and int32 counts")
    import triton

    from .site_gather_kernels import count_edges, prefix_rows, scatter_edges

    degrees = torch.zeros(n_sites, device=index.device, dtype=torch.int32)
    rowptr = torch.empty(n_sites + 1, device=index.device, dtype=torch.int32)
    cursor = torch.empty_like(degrees)
    edges = torch.empty(capacity, device=index.device, dtype=torch.int32)
    if atoms:
        count_edges[(triton.cdiv(capacity, 256),)](
            index,
            count,
            degrees,
            atoms,
            support_capacity,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    prefix_rows[(1,)](
        degrees,
        rowptr,
        cursor,
        n_sites,
        triton.next_power_of_2(n_sites),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if atoms:
        scatter_edges[(triton.cdiv(capacity, 256),)](
            index,
            count,
            rowptr,
            cursor,
            edges,
            atoms,
            support_capacity,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return rowptr, edges


class _Gather(_Direct):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe, recompute_phi):
        if len(p) * recipe.support_capacity > INT32_MAX:
            raise ValueError("CSR edge capacity must fit signed int32")
        import triton

        from .site_gather_kernels import h_forward, overflow_output, owner_output

        x, source, amp, damp, floors, sides = prepare_ids(
            x, p, kernel, charts, recipe, index_dtype=torch.int16
        )
        # H is necessary for output ownership even when dP is not requested.
        # Only dP backward retains it; CSR itself is strictly forward scratch.
        h = x.new_empty((len(p), len(x)))
        y = x.new_zeros((len(x), len(sides[1][0])))
        if len(p):
            pb = triton.next_power_of_2(len(x))
            h_forward[(triton.cdiv(len(p), recipe.atom_group),)](
                x,
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                h,
                len(x),
                x.shape[1],
                recipe.support_capacity,
                pb,
                recipe.support_tile,
                len(p),
                recipe.atom_group,
                floors[0],
                num_warps=4,
                enable_fp_fusion=False,
            )
            rowptr, edges = build_output_csr(
                sides[1][4],
                sides[1][7],
                y.shape[1],
                support_capacity=recipe.support_capacity,
            )
            owner_output[(triton.cdiv(y.shape[1], recipe.site_group),)](
                h,
                amp,
                *[sides[1][i] for i in (0, 1, 3, 6)],
                rowptr,
                edges,
                y,
                len(x),
                y.shape[1],
                pb,
                recipe.support_tile,
                recipe.site_group,
                floors[1],
                num_warps=4,
                enable_fp_fusion=False,
            )
            # Ordered after owner stores: overflow contributions cannot be lost.
            overflow_output[(triton.cdiv(len(p), recipe.atom_group),)](
                h,
                amp,
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                y,
                len(x),
                y.shape[1],
                recipe.support_capacity,
                pb,
                recipe.support_tile,
                len(p),
                recipe.atom_group,
                floors[1],
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.recipe, ctx.floors, ctx.recompute_phi = recipe, floors, True
        saved_h = h if ctx.needs_input_grad[1] else x.new_empty(0)
        ctx.save_for_backward(x, source, amp, damp, saved_h, *sides[0], *sides[1])
        return y


def site_gather_linear(x, p, kernel, charts, recipe):
    return _Gather.apply(x, p, kernel, charts, recipe, True)
