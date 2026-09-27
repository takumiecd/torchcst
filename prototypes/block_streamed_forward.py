"""Forward-only bounded-memory mapped CST materialization and GEMM."""

import math

import torch

from prototypes.block_materialize_kernel import materialize_logical
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


def streamed_forward(layer, x, *, prepared=None, weight_chunk_rows=1024):
    """Generate a row window of logical W and multiply it before reusing the buffer."""
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64:
        raise ValueError("streamed prototype currently requires 64x64 full row tiles")
    if (
        type(weight_chunk_rows) is not int
        or weight_chunk_rows < 64
        or weight_chunk_rows % 64
    ):
        raise ValueError("weight_chunk_rows must be a positive multiple of 64")
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
    if not flat.shape[0]:
        return y.reshape(*x.shape[:-1], layer.shape[0])
    chunk = min(weight_chunk_rows, layer.shape[0])
    w = flat.new_empty((chunk, layer.shape[1]))
    for start in range(0, layer.shape[0], chunk):
        rows = min(chunk, layer.shape[0] - start)
        materialize_logical[(math.ceil(rows / 32), layer.column_groups, 2)](
            p,
            circle,
            section,
            offsets,
            w,
            N=layer.shape[0],
            K=layer.shape[1],
            S=64,
            T=64,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BN=32,
            BK=32,
            BA=1,
            FACTORED=True,
            ROW_GROUP_START=start // 64,
            LOCAL_W=True,
            num_warps=4,
            enable_fp_fusion=True,
        )
        torch.mm(flat, w[:rows].T, out=y[:, start : start + rows])
    return y.reshape(*x.shape[:-1], layer.shape[0])
