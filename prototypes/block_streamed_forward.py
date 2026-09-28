"""Forward-only bounded-memory mapped CST materialization and GEMM."""

import math

import torch

from prototypes.block_materialize_kernel import materialize_logical
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


def streamed_forward(
    layer,
    x,
    *,
    prepared=None,
    weight_chunk_rows=1024,
    materialize_tile=(64, 32),
    cache_weight_rows=0,
    return_cache=False,
):
    """Generate a row window of logical W and multiply it before reusing the buffer."""
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64:
        raise ValueError("streamed prototype currently requires 64x64 full row tiles")
    if (
        type(weight_chunk_rows) is not int
        or weight_chunk_rows < 64
        or weight_chunk_rows % 64
    ):
        raise ValueError("weight_chunk_rows must be a positive multiple of 64")
    if materialize_tile not in ((16, 32), (32, 32), (64, 32), (32, 64), (64, 64)):
        raise ValueError("unsupported materialization tile")
    if (
        type(cache_weight_rows) is not int
        or cache_weight_rows < 0
        or cache_weight_rows >= layer.shape[0]
        or cache_weight_rows % weight_chunk_rows
    ):
        raise ValueError("cache_weight_rows must contain complete windows below full W")
    if torch.is_grad_enabled() and (
        x.requires_grad or layer.strip.atoms.p.requires_grad
    ):
        raise NotImplementedError("mapped streamed prototype supports forward only")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("mapped streamed prototype requires NVIDIA CUDA float32")
    if x.device != layer.strip.atoms.p.device or layer.strip.atoms.p.dtype != x.dtype:
        raise ValueError("input and model must share device and dtype")
    p, circle, section, offsets = (
        prepared
        if prepared is not None
        else prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    )
    if p.shape[1] != 6:
        raise ValueError("streamed factored prototype requires four-dimensional sites")
    flat = x.reshape(-1, layer.shape[1]).contiguous()
    y = flat.new_empty((flat.shape[0], layer.shape[0]))
    cached = flat.new_empty((cache_weight_rows, layer.shape[1]))
    if not flat.shape[0]:
        output = y.reshape(*x.shape[:-1], layer.shape[0])
        return (output, cached) if return_cache else output
    chunk = min(weight_chunk_rows, layer.shape[0])
    bn, bk = materialize_tile
    w = flat.new_empty((chunk, layer.shape[1]))
    for start in range(0, layer.shape[0], chunk):
        rows = min(chunk, layer.shape[0] - start)
        target = cached[start : start + rows] if start < cache_weight_rows else w[:rows]
        materialize_logical[(math.ceil(rows / bn), layer.column_groups, 64 // bk)](
            p,
            circle,
            section,
            offsets,
            target,
            N=layer.shape[0],
            K=layer.shape[1],
            S=64,
            T=64,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BN=bn,
            BK=bk,
            BA=1,
            FACTORED=True,
            ROW_GROUP_START=start // 64,
            LOCAL_W=True,
            num_warps=4,
            enable_fp_fusion=True,
        )
        torch.mm(flat, target.T, out=y[:, start : start + rows])
    output = y.reshape(*x.shape[:-1], layer.shape[0])
    return (output, cached) if return_cache else output
