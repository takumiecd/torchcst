"""Capture exact derivative contractions without rebuilding AD on every update."""

import torch
from torch.func import grad, hessian, jacfwd, jvp, vmap

from ._hessian import hessian_from_gradient
from .factored_taylor import factor_derivatives  # noqa: F401


def local_derivatives(materialize, point):
    def one(p):
        return materialize(p.unsqueeze(0))[0]

    first = jacfwd(one)
    return vmap(first)(point), vmap(jacfwd(first))(point)


def frame_transport(materialize, point, source_point, source_displacement, alpha):
    def represented(p):
        return materialize(p).sum(0)

    def first(p):
        return jvp(represented, (p,), (alpha,))[1]

    visible = (
        first(source_point) + jvp(first, (source_point,), (source_displacement,))[1]
    )
    constant = grad(lambda p: (represented(p) * visible).sum())(point)

    def scalar(p):
        return (materialize(p.unsqueeze(0))[0] * visible).sum()

    blocks = vmap(hessian(scalar))(point)
    return constant, blocks


_GRAPHS = {}


def call(name, materialize, *args):
    from torchcst._runtime.graphs import CapturedCall

    owner = getattr(materialize, "__self__", None)
    if not isinstance(owner, torch.nn.Module):
        return tuple(x.detach() for x in globals()[name](materialize, *args))
    buffers = tuple(
        (n, id(b), b._version, b.device, b.dtype, b.shape)
        for n, b in owner.named_buffers()
    )
    modules = tuple((n, id(m)) for n, m in owner.named_modules())
    key = (name, materialize, buffers, modules)
    if key not in _GRAPHS:
        function = globals()[name]
        # The AD transforms run only while warming/capturing the CUDA graph.
        # Replay performs their exact tensor operations without Python AD work.
        _GRAPHS[key] = CapturedCall(lambda *xs: function(materialize, *xs))
    return _GRAPHS[key](*args)


def factor_observation(factor_atoms, point, inputs, output_gradient):
    def scalar(p):
        u, v = factor_atoms(p.unsqueeze(0))
        return ((inputs @ u) * (output_gradient @ v)).sum()

    return vmap(grad(scalar))(point), vmap(hessian(scalar))(point)


def factor_jgh_observation(factor_atoms, point, inputs, output_gradient):
    """Return current ``jg`` and local ``gh`` using gradient JVPs."""

    def scalar(p):
        u, v = factor_atoms(p.unsqueeze(0))
        return ((inputs @ u) * (output_gradient @ v)).sum()

    gradient = grad(scalar)
    directions = torch.eye(point.shape[-1], device=point.device, dtype=point.dtype)

    def one(parameter_point):
        return gradient(parameter_point), hessian_from_gradient(
            gradient,
            parameter_point,
            directions=directions,
        )

    return vmap(one)(point)
