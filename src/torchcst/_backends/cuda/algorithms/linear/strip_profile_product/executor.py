"""Whole-chart normalization once, small-core tile contractions and VJPs."""

import torch


def strip_product(x, p, value, chart, declaration, recipe):
    import triton as tr

    from ..local_product.contract import Domain
    from ..local_product.executor import local_h
    from ..local_product.preparation import polar_scalars
    from ..profile_product.kernels import prepare
    from .kernels import restrict_tile

    n, out = declaration.shape[1], declaration.shape[0]
    tile = declaration.tile_shape[1]
    local_recipe = recipe.local_recipe()
    scalars = polar_scalars(value)
    packed = p.new_empty((13, len(p)))
    # No autograd through metadata: local VJP consumes the global norm derivative.
    if len(p):
        prepare[(len(p),)](
            p,
            packed,
            len(p),
            n,
            out,
            declaration.axes[1].spacing[0],
            declaration.axes[1].start[0],
            declaration.axes[0].start[0],
            tr.next_power_of_2(n),
            tr.next_power_of_2(out),
            scalars,
            value.spec.normalization.floor,
            BOUNDS=True,
            STRIP_TILE=tile,
            Pitch=chart.tile_pitch,
            num_warps=4,
            enable_fp_fusion=False,
        )
    results = []
    for tile_id, start in enumerate(range(0, n, tile)):
        count = min(tile, n - start)
        restricted = p.new_empty((13, len(p)))
        if len(p):
            restrict_tile[(tr.cdiv(len(p), 256),)](
                packed,
                restricted,
                chart.tile_pitch,
                len(p),
                tile,
                tile_id,
                count,
                out,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        domain = Domain(
            input_size=tile,
            output_size=out,
            input_count=count,
            spacing=declaration.axes[1].spacing[0],
            input_origin=declaration.axes[1].start[0],
            output_origin=declaration.axes[0].start[0],
        )
        ordered = local_recipe.ordered_layout
        results.append(
            local_h(
                x[:, start : start + count],
                p,
                value,
                domain,
                recipe=local_recipe,
                fused_polar=True,
                sparse=ordered,
                hybrid=ordered,
                three_band=ordered,
                singletons=ordered,
                tile_packed=ordered,
                product_floor=value.spec.normalization.floor,
                prepared=restricted if ordered else restricted[:9].contiguous(),
            )
        )
    # Each input tile contributes to the same output; no tile-local renormalization.
    return torch.stack(results).sum(0)
