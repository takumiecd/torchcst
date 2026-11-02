"""Torch reference for a radial atom over (output_chart, input_chart).

q_oi,a = d_out(o,c_out,a)^2 / sigma_out,a^2
       + d_in(i,c_in,a)^2 / sigma_in,a^2.
The profile is evaluated once and L2-normalized over all (o,i) sites.
"""

import torch

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.kernels.profiles import GaussianSpec


def materialize_atoms(state, output_chart, input_chart, p):
    from .execution import family

    name = family(state)
    profile = state.profiles[0]
    if name == "radial":
        centers = p
        amplitude = p.new_ones(p.shape[0])
        pin = pout = profile.sigma.to(p).reciprocal().square()
    elif name in ("direct_amp_width", "polar_amp_width"):
        from torchcst._backends.torch.parameterizations import (
            direct_amp_width,
            polar_amp_width,
        )

        law = polar_amp_width if name == "polar_amp_width" else direct_amp_width
        amplitude, alpha = law._amplitude_and_alpha(state, p[:, :2])
        sigma_in, sigma_out = law._bandwidth_sigmas(state, amplitude, alpha)
        pin, pout = (s.reciprocal().square().detach() for s in (sigma_in, sigma_out))
        centers = p[:, 2:]
    else:
        raise ValueError("unsupported radial pair parameterization")
    split = output_chart.center_parameter_dim
    qo = _charts.squared_distance(output_chart, centers[:, :split]) * pout
    qi = _charts.squared_distance(input_chart, centers[:, split:]) * pin
    q = (qo[:, None, :] + qi[None, :, :]).reshape(
        output_chart.ncs * input_chart.ncs, p.shape[0]
    )
    normalized = profile.binding.normalization.kind == "discrete_l2"
    if type(profile.binding.profile) is GaussianSpec and normalized:
        # Validate shape/revision before the stable Gaussian specialization.
        _profile._module(profile)
        values = torch.exp(0.5 * torch.log_softmax(-q, dim=0))
    else:
        values = _profile.shape_function(
            profile, "unnormalized_from_squared", q, p.new_tensor(1.0)
        )
        if normalized:
            norm = torch.linalg.vector_norm(values, dim=0)
            values = values / norm.clamp_min(profile.binding.normalization.floor)
    return (values * amplitude).T.reshape(p.shape[0], output_chart.ncs, input_chart.ncs)
