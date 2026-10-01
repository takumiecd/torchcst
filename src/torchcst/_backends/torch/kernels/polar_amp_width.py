"""PyTorch execution for legacy polar_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart
from torchcst.kernels.base import AtomInit


def lower_half_amplitude(self) -> Tensor:
    """Absolute amplitude where ``L(w)`` is halfway between its limits."""

    return self.w_c / self.lower_kappa.sqrt()


def initialize(
    self,
    input_chart: Chart,
    output_chart: Chart,
    atoms: int,
    *,
    mode: AtomInit,
) -> Tensor:
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
    output_p = self.profile.initialize(output_chart, atoms, mode=mode)

    amplitude = input_p.new_empty(atoms)
    amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    maximum = self.amplitude_max.to(amplitude)
    # Keep initialization away from the angular critical points w = +/- W.
    ratio = (amplitude / maximum).clamp(-1 + 1e-6, 1 - 1e-6)
    polar = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
    # Preserve the initialized angular amplitude while giving the
    # bandwidth clock an optional, regularized exploration reserve.
    initial_radius = (1.0 + 3.0 * self.alpha_init.to(polar)).sqrt()
    polar = polar * initial_radius
    return torch.cat((polar, input_p, output_p), dim=-1)


def materialize_atoms(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> Tensor:
    phi_input, phi_output = self.factors(input_chart, output_chart, p)
    return torch.einsum("oa,ia->aoi", phi_output, phi_input)


def factors(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor]:
    polar, input_p, output_p = self._split(input_chart, output_chart, p)
    amplitude, alpha = self._amplitude_and_alpha(polar)
    sigma_input, sigma_output = self._bandwidth_sigmas(amplitude, alpha)
    # Width is a state derived from update history, not a task-loss degree
    # of freedom. Only the explicit radial regularizer may decrease alpha.
    precision_input = sigma_input.reciprocal().square().detach()
    precision_output = sigma_output.reciprocal().square().detach()
    phi_input = self.profile.evaluate_with_precision(
        input_chart, input_p, precision_input
    )
    phi_output = self.profile.evaluate_with_precision(
        output_chart, output_p, precision_output
    )
    return phi_input, phi_output * amplitude.unsqueeze(0)
