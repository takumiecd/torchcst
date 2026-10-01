"""Linear execution policies, independent of module and optimizer ownership.

Triton is imported only when its explicit backend executes. The automatic
policy stays conservative until GPU cost measurements justify changing it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from torch import Tensor

from torchcst.geometry import Chart
from torchcst.kernels.state import KernelState

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear

Backend = Literal["auto", "factored", "materialized", "tiled", "triton"]
ResolvedBackend = Literal["factored", "materialized", "tiled", "triton"]


@dataclass(frozen=True)
class _Implementation:
    validate: Callable[[tuple[Chart, ...], KernelState], None]
    forward: Callable[[CSTLinear, Tensor, Tensor], Tensor]


def _validate_triton(charts: tuple[Chart, ...], kernel: KernelState) -> None:
    from torchcst._backends.torch.operators.strip_torus.preparation import validate

    validate(charts, kernel)


def _triton_forward(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    from torchcst._backends.cuda.algorithms.strip_torus.fused.executor import forward

    return forward(site, inputs, p)


def _torch_call(name, *args):
    from torchcst._backends.torch.operators import dispatch

    return getattr(dispatch, name)(*args)


_IMPLEMENTATIONS = {
    name: _Implementation(
        partial(_torch_call, "validate_" + name), partial(_torch_call, name)
    )
    for name in ("materialized", "factored", "tiled")
}
_IMPLEMENTATIONS["triton"] = _Implementation(_validate_triton, _triton_forward)


def validate_backend(
    name: Backend, charts: tuple[Chart, ...], kernel: KernelState
) -> None:
    if name == "auto":
        return
    if name not in _IMPLEMENTATIONS:
        choices = ", ".join(repr(key) for key in ("auto", *_IMPLEMENTATIONS))
        raise ValueError(f"backend must be one of {choices}")
    _IMPLEMENTATIONS[name].validate(charts, kernel)


def resolve_backend(site: CSTLinear) -> ResolvedBackend:
    from torchcst._backends.torch.kernels import execution as _kernel

    if site.backend != "auto":
        return site.backend
    if len(site.cst_charts()) == 1 or not _kernel.supports_factorization(site.kernel):
        return "materialized"
    factor_size = site.atom_count * (site.in_features + site.out_features)
    dense_size = site.in_features * site.out_features
    return "factored" if factor_size <= dense_size else "materialized"


def linear_forward(
    site: CSTLinear, inputs: Tensor, p: Tensor, *, backend: ResolvedBackend
) -> Tensor:
    # CSTLinear validates backend assignments; execution need not synchronize
    # chart configuration buffers with the host on every dispatch.
    return _IMPLEMENTATIONS[backend].forward(site, inputs, p)
