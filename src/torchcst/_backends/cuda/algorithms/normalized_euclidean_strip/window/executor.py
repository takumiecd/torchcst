"""Bounded GEMM orchestration with a rolling eight-row derivative halo.

Provider: prepare(p)->state; build(p,state,start,count,out); vjp(p,state,
start,dw,previous8,dp). vjp processes owner=clamp((nearest_row+4)//R).
It must preserve the original full-support point reduction, reading earlier
rows from previous8 and current rows from dw. No raw-minus-coupling shortcut.
"""

import torch


class _WindowApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, provider, rows, window):
        if window < 9:
            raise ValueError("rolling radius-four halo requires window >= 9")
        state = provider.prepare(p)
        work = x.new_empty((min(window, rows), x.shape[1]))
        y = x.new_empty((x.shape[0], rows))
        for start in range(0, rows, window):
            count = min(window, rows - start)
            tile = work[:count]
            provider.build(p, state, start, count, tile)
            torch.mm(x, tile.T, out=y[:, start : start + count])
        ctx.save_for_backward(x, p)
        ctx.work = work
        ctx.provider, ctx.state, ctx.rows, ctx.window = provider, state, rows, window
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, p = ctx.saved_tensors
        work = ctx.work
        provider, state, rows, window = ctx.provider, ctx.state, ctx.rows, ctx.window
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dp = torch.zeros_like(p) if ctx.needs_input_grad[1] else None
        halo = x.new_empty((8, x.shape[1])) if dp is not None else None
        for start in range(0, rows, window):
            count = min(window, rows - start)
            tile = work[:count]
            dys = dy[:, start : start + count]
            if dx is not None:
                provider.build(p, state, start, count, tile)
                torch.addmm(dx, dys, tile, beta=0 if start == 0 else 1, out=dx)
            if dp is not None:
                torch.mm(dys.T, x, out=tile)
                provider.vjp(p, state, start, tile, halo, dp)
                # All nonfinal windows have >=9 rows; final halo is never read.
                if start + count < rows:
                    halo.copy_(tile[-8:])
        return dx, dp, None, None, None


def window_linear(x, params, provider, *, rows, window=512):
    return _WindowApply.apply(x, params, provider, rows, window)


def scratch_bytes(rows, columns, window=512):
    """FP32 work+rolling halo, excluding state/output/vendor workspace."""
    return 4 * columns * (min(rows, window) + 8)


def execute_window(*, x, parameters, operator, recipe):
    from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.window.provider import (
        PackedWindowProvider,
    )

    from .._shared.geometry import _geometry

    rows = min(recipe.window_rows, operator.out_features)
    return window_linear(
        x,
        parameters,
        PackedWindowProvider(
            _geometry(operator, x.device), rows, recipe.enable_fp_fusion
        ),
        rows=operator.out_features,
        window=rows,
    )
