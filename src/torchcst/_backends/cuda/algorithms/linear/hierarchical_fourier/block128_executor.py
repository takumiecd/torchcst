"""Lazy CUDA C++ extension; metadata/CPU imports never load a compiler.

One CTA owns 4 batch rows x 32 output sites. One thread contracts one atom.
H/G parameter fields remain thread-local; only H/G values cross threads.
"""

import hashlib
from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from .contract import Block128FourierRecipe, FourierLinearSpec


@cache
def extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [directory / "block128_bindings.cpp", directory / "block128_kernels.cu"]
    identity = hashlib.sha256()
    for path in [*sources, directory / "block128_math.h"]:
        identity.update(path.name.encode())
        identity.update(path.read_bytes())
    return load(
        name="torchcst_block128_" + identity.hexdigest()[:16],
        sources=[str(path) for path in sources],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--fmad=false", "-lineinfo"],
    )


@cache
def fit_table(device):
    from .executor import coefficient_table

    table, lower = coefficient_table(8)
    if table.shape != (32, 9) or lower != 0.29:
        raise RuntimeError("block128 coefficient table contract changed")
    return table.to(device=device).contiguous()


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, fit):
        y, norm = extension().forward(x, p, fit)
        # Keep original tensors, not detached copies: retained backward must
        # still detect edits to X/P made after this forward.
        ctx.save_for_backward(x, p, fit, norm)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, p, fit, norm = ctx.saved_tensors
        dx, dp = extension().backward(x, p, dy.contiguous(), fit, norm)
        return dx, dp, None


def apply(x, p, spec, recipe):
    if type(recipe) is not Block128FourierRecipe:
        raise TypeError("requires Block128FourierRecipe")
    if type(spec) is not FourierLinearSpec or spec.size != 128 or spec.modes != 8:
        raise ValueError("block128 requires FourierLinearSpec(128, modes=8)")
    if (
        x.device.type != "cuda"
        or p.device != x.device
        or x.dtype != torch.float32
        or p.dtype != torch.float32
        or x.ndim != 2
        or x.shape[1] != 128
        or not 1 <= x.shape[0] <= 64
        or p.shape != (128, 4)
        or not x.is_contiguous()
        or not p.is_contiguous()
    ):
        raise ValueError(
            "block128 requires contiguous CUDA FP32 X[B1..64,128], P[128,4]"
        )
    return _Apply.apply(x, p, fit_table(x.device))
