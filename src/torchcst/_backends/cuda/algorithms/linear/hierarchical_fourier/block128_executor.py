"""Lazy CUDA C++ extension; metadata/CPU imports never load a compiler.

One CTA owns 4 batch rows x 32 output sites. One thread contracts one atom.
H/G parameter fields remain thread-local; only H/G values cross threads.
"""

import hashlib
from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from .contract import (
    Block128FourierRecipe,
    FourierLinearSpec,
    SortedBlock128FourierRecipe,
)


@cache
def extension():
    """最初の実行でだけC++/CUDAをコンパイルする。宣言のimportでは呼ばない。

    headerを含むsourceのhashを拡張名に入れる。コードを変更したときに古いbinaryを
    使い回さないためであり、モデルやatomの値のhashではない。
    """
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [directory / "block128_bindings.cpp", directory / "block128_kernels.cu"]
    identity = hashlib.sha256()
    for path in [
        *sources,
        directory / "block128_math.h",
        directory / "block128_layout.h",
    ]:
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
    """幅や中心に依存しない係数表を、一度だけ対象deviceへ転送する。"""
    from .executor import coefficient_table

    table, lower = coefficient_table(8)
    if table.shape != (32, 9) or lower != 0.29:
        raise RuntimeError("block128 coefficient table contract changed")
    return table.to(device=device).contiguous()


class _Apply(torch.autograd.Function):
    """比較基準：ソートせず、全128 atomを対象にする元の経路。"""
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


class _SortedApply(torch.autograd.Function):
    """ソート済みコピーの寿命を一つのforward/backwardに閉じる。

    元のparametersは任意順のまま。CUDAが元の行番号を保存し、backwardの最後に
    その行番号へ勾配を戻す。optimizerのmomentを並べ替える必要はない。
    """

    @staticmethod
    def forward(ctx, x, parameters, fit):
        (
            output,
            norm,
            sorted_parameters,
            source_rows,
            atom_groups,
            ranges,
        ) = extension().forward_sorted(x, parameters, fit)

        # 元Tensorも保存する。コピーだけでは、forward後のin-place変更を検出できない。
        # layoutはretained backwardでも同じものを使い、次の呼出しへ上書きしない。
        ctx.save_for_backward(
            x,
            parameters,
            fit,
            norm,
            sorted_parameters,
            source_rows,
            atom_groups,
            ranges,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (
            x,
            _original_parameters,
            fit,
            norm,
            sorted_parameters,
            source_rows,
            atom_groups,
            ranges,
        ) = ctx.saved_tensors

        # _original_parametersは、上のsaved_tensors取得時のversion検査のためにも必要。
        dx, parameter_gradients = extension().backward_sorted(
            x,
            sorted_parameters,
            dy.contiguous(),
            fit,
            norm,
            source_rows,
            atom_groups,
            ranges,
        )
        return dx, parameter_gradients, None


def apply(x, p, spec, recipe):
    """128版の入口。形を検査してから、選ばれたautograd実装へ渡す。"""
    if type(recipe) not in (Block128FourierRecipe, SortedBlock128FourierRecipe):
        raise TypeError("requires a block128 recipe")
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
    fit = fit_table(x.device)

    if type(recipe) is SortedBlock128FourierRecipe:
        return _SortedApply.apply(x, p, fit)

    return _Apply.apply(x, p, fit)
