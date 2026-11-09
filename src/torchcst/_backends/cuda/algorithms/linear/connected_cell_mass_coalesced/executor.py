"""Coalesced rows and shared moments, first-order VJP."""

import hashlib
from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@cache
def extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    src = directory / "pipeline.cu"
    inputs = [
        src,
        directory.parent / "connected_cell_mass_reuse/profile_m16.cuh",
        directory.parent / "radius_cell_mass/common.cuh",
        directory.parent / "radius_cell_mass/limits.cuh",
    ]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:12]
    return load(
        name=f"torchcst_connected_coalesced_{digest}",
        sources=[str(src)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--ftz=false"],
        verbose=False,
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, ni, no, stage):
        y, *saved = extension().forward(x, p, ni, no, 16, True, [0.1, 0.9, 1.25], stage)
        ctx.save_for_backward(x, p, *saved)
        ctx.config = ni, no, stage
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        ni, no, stage = ctx.config
        dx, dp = extension().backward(
            *ctx.saved_tensors[:2],
            dy.contiguous(),
            *ctx.saved_tensors[2:],
            ni,
            no,
            16,
            True,
            [0.1, 0.9, 1.25],
            stage,
        )
        return dx, dp, None, None, None


def linear(x, p, layout, recipe):
    (no, co), (ni, ci) = layout
    if ci != co:
        raise ValueError("requires identical connection declarations")
    return _Linear.apply(x, p, ni, no, 0)
