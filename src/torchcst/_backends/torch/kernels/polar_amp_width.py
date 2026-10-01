"""PyTorch execution for declared polar_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.parameterizations import (
    polar_amp_width as _parameterizations,
)
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState
from torchcst.kernels.spec import AtomInit


def lower_half_amplitude(state) -> Tensor:
    """Absolute amplitude where ``L(w)`` is halfway between its limits."""

    return state.scalar("w_c") / state.scalar("lower_kappa").sqrt()


def initialize(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    atoms: int,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    input_p = _profile.initialize(state.profiles[0], input_chart, atoms, mode="uniform")
    output_p = _profile.initialize(state.profiles[1], output_chart, atoms, mode=mode)

    amplitude = input_p.new_empty(atoms)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    maximum = state.scalar("amplitude_max").to(amplitude)
    # Keep initialization away from the angular critical points w = +/- W.
    ratio = (amplitude / maximum).clamp(-1 + 1e-6, 1 - 1e-6)
    polar = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
    # Preserve the initialized angular amplitude while giving the
    # bandwidth clock an optional, regularized exploration reserve.
    initial_radius = (1.0 + 3.0 * state.scalar("alpha_init").to(polar)).sqrt()
    polar = polar * initial_radius
    return torch.cat((polar, input_p, output_p), dim=-1)


def materialize_atoms(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> Tensor:
    phi_input, phi_output = _kernel.factors(state, input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    polar, input_p, output_p = _parameterizations._split(
        state, input_chart, output_chart, p
    )
    amplitude, alpha = _parameterizations._amplitude_and_alpha(state, polar)
    sigma_input, sigma_output = _parameterizations._bandwidth_sigmas(
        state, amplitude, alpha
    )
    # Width is a state derived from update history, not a task-loss degree
    # of freedom. Only the explicit radial regularizer may decrease alpha.
    precision_input = sigma_input.reciprocal().square().detach()
    precision_output = sigma_output.reciprocal().square().detach()
    phi_input = _profile.evaluate_with_precision(
        state.profiles[0], input_chart, input_p, precision_input
    )
    phi_output = _profile.evaluate_with_precision(
        state.profiles[1], output_chart, output_p, precision_output
    )
    return phi_input, phi_output * amplitude.unsqueeze(0)
