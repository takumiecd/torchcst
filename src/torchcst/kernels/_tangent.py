"""Kernel-owned coordinate interpretation for exact first factor derivatives."""

import torch


def separable(kernel, input_chart, output_chart, p):
    width = kernel.parameter_dim(input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != width:
        raise ValueError("parameter shape does not match separable kernel")
    ni = kernel.input_profile.parameter_dim(input_chart)
    u, du = kernel.input_profile.tangent(input_chart, p[:, :ni])
    v, dv = kernel.output_profile.tangent(output_chart, p[:, ni:])
    ju = p.new_zeros(p.shape[0], input_chart.features, width)
    jv = p.new_zeros(p.shape[0], output_chart.features, width)
    ju[:, :, :ni] = du.permute(1, 0, 2)
    jv[:, :, ni:] = dv.permute(1, 0, 2)
    return u.T, v.T, ju, jv


def amplitude(kernel, inner, input_chart, output_chart, p):
    a, coordinates = kernel._split(input_chart, output_chart, p)
    u, v, ju, jv = inner(coordinates)
    du = torch.cat((torch.zeros_like(u[..., None]), ju), -1)
    dv = torch.cat((v[..., None], a[..., None] * jv), -1)
    return u, a * v, du, dv


def amplitude_bandwidth(kernel, input_chart, output_chart, p):
    a, source, target = kernel._split(input_chart, output_chart, p)
    precision = kernel.bandwidth_precision(input_chart, output_chart, p)
    u, du, dku = kernel.profile.tangent_with_precision(
        input_chart, source, precision
    )
    v, dv, dkv = kernel.profile.tangent_with_precision(
        output_chart, target, precision
    )
    gate = kernel.amplitude_gate(input_chart, output_chart, p)
    narrow = kernel.sigma_min.reciprocal().square()
    broad = kernel.sigma_max.reciprocal().square()
    dprecision = (
        (narrow - broad)
        * gate
        * (1 - gate)
        * (
            2
            * a[:, 0]
            / (kernel.temperature * (a[:, 0].square() + kernel.gate_eps.square()))
        )
    )
    ni = source.shape[-1]
    ju = p.new_zeros(p.shape[0], input_chart.features, p.shape[1])
    jv = p.new_zeros(p.shape[0], output_chart.features, p.shape[1])
    ju[:, :, 0] = (dku * dprecision[None]).T
    ju[:, :, 1 : 1 + ni] = du.permute(1, 0, 2)
    jv[:, :, 0] = (v + a.T * dkv * dprecision[None]).T
    jv[:, :, 1 + ni :] = a[..., None] * dv.permute(1, 0, 2)
    return u.T, a * v.T, ju, jv
