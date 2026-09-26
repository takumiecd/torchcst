"""Local atom contractions using I/B packing; no weight tiles or fine remap.

Experimental entry points, not a public backend. The Torch oracle uses
canonical profile evaluation. The CUDA implementation has a direct backward.
"""

import torch
from torch.autograd.function import once_differentiable

from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan, prepare
from torchcst.nn._strip_torus import validate_tiled
from torchcst.nn._support_layout import station_buckets, support_layout


def reference(site, inputs, p):
    validate_tiled(site.chart, site.kernel)
    plan = execution_plan(site)
    encoded, amplitude, precision = site.kernel.tile_parameters(site.chart, p)
    layout = support_layout(
        plan,
        site.chart.geometry.decode_centers(encoded),
        precision,
        site.chart.tile_shape[0],
    )
    n, k = site.chart.shape
    flat = inputs.reshape(-1, k)
    columns = torch.arange(k, device=p.device)
    outputs = []
    for row in range(n):
        station = row // site.chart.tile_shape[0]
        result = flat.sum(-1) * 0 + p.sum() * 0
        for bucket in station_buckets(station, site.chart.tile_count):
            for a in layout.order[
                layout.offsets[bucket] : layout.offsets[bucket + 1]
            ].tolist():
                selection = row * k + columns
                squared = site.chart.squared_distance(
                    encoded[a : a + 1], selection
                ).squeeze(-1)
                selected = columns[(squared.detach() * precision[a]) < 1]
                values = site.kernel.profile.evaluate_with_precision_slice(
                    site.chart,
                    encoded[a : a + 1],
                    precision[a : a + 1],
                    row * k + selected,
                ).squeeze(-1)
                result = result + (flat[:, selected] * values).sum(-1) * amplitude[a]
        outputs.append(result)
    return torch.stack(outputs, -1).reshape(*inputs.shape[:-1], n)


def forward(site, inputs, p, *, batch_tile=4, column_tile=128):
    if inputs.device.type != "cuda" or torch.version.hip or p.device != inputs.device:
        raise ValueError("local atom prototype requires NVIDIA CUDA")
    if inputs.dtype != torch.float32 or p.dtype != torch.float32:
        raise TypeError("local atom prototype requires float32")
    if site.chart.device != p.device or site.chart.dtype != p.dtype:
        raise ValueError("chart and atoms must share dtype and device")
    if batch_tile not in (1, 4, 8, 16) or column_tile not in (32, 64, 128, 256):
        raise ValueError("unsupported local execution tile")
    packed, circle, section, offsets = prepare(site, p, support_layout=True)
    bounds = torch.stack((section.amin(0), section.amax(0)))
    x = inputs.reshape(-1, site.in_features).contiguous()
    result = LocalAtom.apply(
        x,
        packed,
        circle,
        section,
        offsets,
        bounds,
        site.chart.tile_shape[0],
        PROFILE_KINDS[type(site.kernel.profile)],
        batch_tile,
        column_tile,
    )
    return result.reshape(*inputs.shape[:-1], site.out_features)


class LocalAtom(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        packed,
        circle,
        section,
        offsets,
        bounds,
        station_rows,
        profile,
        bm=4,
        bk=128,
    ):
        from prototypes.local_atom_kernels import direct_forward

        n, k = circle.shape[0], section.shape[0]
        options = {
            "M": x.shape[0],
            "N": n,
            "K": k,
            "D": packed.shape[1] - 2,
            "G": (n + station_rows - 1) // station_rows,
            "S": station_rows,
            "PROFILE": profile,
            "BM": bm,
            "BK": bk,
        }
        ctx.options = options
        ctx.save_for_backward(x, packed, circle, section, offsets, bounds)
        y = x.new_empty((x.shape[0], n))
        if x.shape[0]:
            with torch.cuda.device(x.device):
                direct_forward[((x.shape[0] + bm - 1) // bm, n)](
                    x,
                    packed,
                    circle,
                    section,
                    offsets,
                    bounds,
                    y,
                    **options,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        import triton as tr

        from prototypes.local_atom_kernels import direct_dp, direct_dx

        x, packed, circle, section, offsets, bounds = ctx.saved_tensors
        opts = ctx.options
        dy = gradient.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dp = torch.zeros_like(packed) if ctx.needs_input_grad[1] else None
        if x.shape[0]:
            with torch.cuda.device(x.device):
                if dx is not None:
                    direct_dx[(tr.cdiv(x.shape[0], opts["BM"]), opts["K"])](
                        dy,
                        packed,
                        circle,
                        section,
                        offsets,
                        dx,
                        **opts,
                        RN=tr.next_power_of_2(opts["S"]),
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                if dp is not None and packed.shape[0]:
                    direct_dp[(packed.shape[0],)](
                        x,
                        dy,
                        packed,
                        circle,
                        section,
                        offsets,
                        bounds,
                        dp,
                        **opts,
                        BG=tr.next_power_of_2(2 * opts["G"] + 1),
                        DD=tr.next_power_of_2(opts["D"]),
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        return (dx, dp) + (None,) * (len(ctx.needs_input_grad) - 2)
