from __future__ import annotations

import pytest
import torch
from kernel_cases import direct_state, polar_state

from torchcst import BandwidthBounds, Chart
from torchcst._backends.torch.kernels import execution as _kernel


def _rows(kernel_type, amplitude, alpha):
    q = 1.0 + 3.0 * alpha
    if kernel_type is polar_state:
        ratio = amplitude / 2.0
        direction = torch.stack(
            (ratio, (1.0 - ratio.square()).clamp_min(0).sqrt()), dim=-1
        )
        head = direction * q.sqrt().unsqueeze(-1)
    else:
        head = torch.stack((amplitude, q), dim=-1)
    return torch.cat(
        (head, torch.zeros(amplitude.numel(), 2, dtype=amplitude.dtype)), -1
    )


@pytest.mark.parametrize("kernel_type", [polar_state, direct_state])
def test_effective_bandwidth_bounds_never_cross_for_split_sides(kernel_type):
    value = kernel_type(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        lower_kappa=0.1,
        upper_decay_power=0.75,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
        output_bounds=BandwidthBounds(
            minimum=0.15, maximum=0.8, birth=0.8, upper_floor=0.15
        ),
    ).double()
    input_chart = Chart.linspace(3, spacing=1.0)
    output_chart = Chart.linspace(4, spacing=1.0)
    amplitude = torch.linspace(0.0, 2.0, 401, dtype=torch.float64)
    for alpha_value in (0.0, 0.5, 1.0):
        alpha = torch.full_like(amplitude, alpha_value)
        p = _rows(kernel_type, amplitude, alpha)
        bounds = _kernel.coordinate(
            value, "bandwidth_bounds_by_side", input_chart, output_chart, p
        )
        sigmas = _kernel.coordinate(
            value, "bandwidth_sigmas", input_chart, output_chart, p
        )
        for (lower, upper), sigma in zip(bounds, sigmas):
            assert torch.all(lower <= upper)
            assert torch.all(lower <= sigma + 1e-14)
            assert torch.all(sigma <= upper + 1e-14)
            if alpha_value == 0.0:
                torch.testing.assert_close(sigma, lower)
            elif alpha_value == 1.0:
                torch.testing.assert_close(sigma, upper)
    crossing = _rows(
        kernel_type,
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    for lower, upper in _kernel.coordinate(
        value, "bandwidth_bounds_by_side", input_chart, output_chart, crossing
    ):
        torch.testing.assert_close(lower, upper)


@pytest.mark.parametrize("kernel_type", [polar_state, direct_state])
def test_sublinear_upper_decay_cannot_cross_near_zero_amplitude(kernel_type):
    value = kernel_type(
        amplitude_max=2.0,
        w_c=0.5,
        upper_decay_power=0.75,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).double()
    input_chart = Chart.linspace(3, spacing=1.0)
    output_chart = Chart.linspace(4, spacing=1.0)
    p = _rows(
        kernel_type,
        torch.tensor([0.001], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    lower, upper = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    torch.testing.assert_close(lower, upper)
