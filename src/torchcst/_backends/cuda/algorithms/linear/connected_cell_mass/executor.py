"""FP32 shared-moment candidate with geometry caches, first-order VJP only."""

import hashlib
from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@cache
def extension():
    from torch.utils.cpp_extension import load

    p = Path(__file__).parent
    src = p / "pipeline.cu"
    headers = sorted((p.parent / "radius_cell_mass").glob("*.cuh"))
    digest = hashlib.sha256(
        b"".join(f.read_bytes() for f in [src, *headers])
    ).hexdigest()[:12]
    return load(
        name=f"torchcst_connected_{digest}",
        sources=[str(src)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--ftz=false"],
        verbose=False,
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, ni, no, m, connected, settings):
        y, *saved = extension().forward(x, p, ni, no, m, connected, settings)
        ctx.save_for_backward(x, p, *saved)
        ctx.config = ni, no, m, connected, settings
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        dx, dp = extension().backward(
            *ctx.saved_tensors[:2], dy.contiguous(), *ctx.saved_tensors[2:], *ctx.config
        )
        return dx, dp, None, None, None, None, None


def linear(x, p, layout, recipe):
    (no, co), (ni, ci) = layout
    if ci != co:
        raise ValueError("first candidate requires equal connection settings")
    settings = (
        (0.1, 0.9, 1.25)
        if ci is None
        else (ci.position_half_width, ci.scale_upper, ci.scale_ratio)
    )
    return _Linear.apply(x, p, ni, no, recipe.modes, recipe.connected, settings)


@cache
def control_extension():
    """Independent macro build; old native guards/defaults stay unchanged."""
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent.parent / "radius_cell_mass"
    source, host = directory / "pipeline.cu", directory / "host_autograd.cpp"
    digest = hashlib.sha256(
        b"".join(
            p.read_bytes() for p in [source, host, *sorted(directory.glob("*.cuh"))]
        )
    ).hexdigest()[:12]
    return load(
        name=f"torchcst_connected_control_{digest}",
        sources=[str(source), str(host)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=[
            "-O2",
            "--ftz=false",
            "-DRADIUS_ROW_BATCH=8",
            "-DRADIUS_CONNECTED_CONTROL=1",
        ],
        verbose=False,
    )
