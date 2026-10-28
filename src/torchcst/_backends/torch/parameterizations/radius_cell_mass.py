"""Public radius diagnostics over physical or existing Polar coordinates."""

import torch

from torchcst.kernels.parameterizations import PhysicalAmpRadiusSpec

from .polar_amp_width import _amplitude_and_alpha, _split, _width_bounds


def decoded_values(state, p):
    if type(state.spec.parameterization) is PhysicalAmpRadiusSpec:
        return p[:, 0].double(), p[:, 1].double(), p[:, 1].double()
    amplitude, alpha = _amplitude_and_alpha(
        state, p[:, :2].double(), tiny=torch.finfo(p.dtype).tiny
    )
    names = state.spec.parameterization.width_scalar_names
    input_radius, _, _ = _width_bounds(
        state, amplitude, alpha, side="input", scalar_names=names
    )
    output_radius, _, _ = _width_bounds(
        state, amplitude, alpha, side="output", scalar_names=names
    )
    return amplitude, input_radius, output_radius


def support_radii(state, input_chart, output_chart, p):
    _split(state, input_chart, output_chart, p)
    _, inputs, outputs = decoded_values(state, p)
    return inputs, outputs


def support_radius(state, input_chart, output_chart, p):
    inputs, outputs = support_radii(state, input_chart, output_chart, p)
    if not torch.equal(inputs, outputs):
        raise ValueError("input/output support radii differ; use support_radii")
    return inputs


def amplitude(state, input_chart, output_chart, p):
    _split(state, input_chart, output_chart, p)
    return decoded_values(state, p)[0]
