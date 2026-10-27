"""Independent cell-integral reference using the existing Polar coordinate law."""

import torch

from torchcst._backends.torch.parameterizations import polar_amp_width as polar

from .hierarchical_cell_mass import _axes, axis_masses
from .polar_amp_width import initialize  # noqa: F401


def decoded_values(state, p):
    # Decode in FP64 for Fourier evaluation, keeping the source singular guard.
    # The Polar law itself and its update policy are unchanged.
    amplitude, alpha = polar._amplitude_and_alpha(
        state, p[:, :2].double(), tiny=torch.finfo(p.dtype).tiny
    )
    sigma_input, sigma_output = polar._bandwidth_sigmas(state, amplitude, alpha)
    return amplitude, sigma_input.detach(), sigma_output.detach()


def factors(state, input_chart, output_chart, p):
    axes = _axes(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != 4 or p.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Polar cell masses require FP32/FP64 [s,t,input_center,output_center]"
        )
    if not bool(torch.isfinite(p).all()):
        raise ValueError("Polar atoms must be finite")
    amplitude, sigma_input, sigma_output = decoded_values(state, p)
    v = axis_masses(axes[0], sigma_input, p[:, 2].double())
    u = axis_masses(axes[1], sigma_output, p[:, 3].double())
    return v.T.to(p.dtype), (u.T * amplitude).to(p.dtype)


def materialize_atoms(state, input_chart, output_chart, p):
    v, u = factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", u, v)


def weight(state, input_chart, output_chart, p):
    result = p.new_zeros((output_chart.features, input_chart.features))
    for begin in range(0, len(p), 512):
        v, u = factors(state, input_chart, output_chart, p[begin : begin + 512])
        result = result + u @ v.T
    return result
