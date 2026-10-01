"""PyTorch coordinate interpretation for exact first factor derivatives."""

import torch

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile

from ..parameterizations import amp_width as _amp_width
from . import amplitude as _amplitude


def separable(kernel, input_chart, output_chart, p):
    width = _kernel.parameter_dim(kernel, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != width:
        raise ValueError("parameter shape does not match separable kernel")
    ni = _profile.parameter_dim(kernel.profiles[0], input_chart)
    u, du = _profile.tangent(kernel.profiles[0], input_chart, p[:, :ni])
    v, dv = _profile.tangent(kernel.profiles[1], output_chart, p[:, ni:])
    ju = p.new_zeros(p.shape[0], input_chart.features, width)
    jv = p.new_zeros(p.shape[0], output_chart.features, width)
    ju[:, :, :ni] = du.permute(1, 0, 2)
    jv[:, :, ni:] = dv.permute(1, 0, 2)
    return u.T, v.T, ju, jv


def amplitude(kernel, inner, input_chart, output_chart, p):
    a, coordinates = _amplitude._split(kernel, input_chart, output_chart, p)
    u, v, ju, jv = inner(coordinates)
    du = torch.cat((torch.zeros_like(u[..., None]), ju), -1)
    dv = torch.cat((v[..., None], a[..., None] * jv), -1)
    return u, a * v, du, dv


def amplitude_bandwidth(kernel, input_chart, output_chart, p):
    a, source, target = _amp_width._split(kernel, input_chart, output_chart, p)
    precision, dprecision = _amp_width._precision_and_jacobian(kernel, a)
    u, du, dku = _profile.tangent_with_precision(
        kernel.profiles[0], input_chart, source, precision
    )
    v, dv, dkv = _profile.tangent_with_precision(
        kernel.profiles[0], output_chart, target, precision
    )
    ni = source.shape[-1]
    ju = p.new_zeros(p.shape[0], input_chart.features, p.shape[1])
    jv = p.new_zeros(p.shape[0], output_chart.features, p.shape[1])
    ju[:, :, 0] = (dku * dprecision[None]).T
    ju[:, :, 1 : 1 + ni] = du.permute(1, 0, 2)
    jv[:, :, 0] = (v + a.T * dkv * dprecision[None]).T
    jv[:, :, 1 + ni :] = a[..., None] * dv.permute(1, 0, 2)
    return u.T, a * v.T, ju, jv
