"""Projected a/q proposals; Adam moments retain their ordinary meaning."""

from functools import lru_cache

import torch


def project_parameter_gradient(state, output_chart, input_chart, p, gradient):
    return gradient


@lru_cache(maxsize=128)
def _rounded_amplitude_max(cap, dtype):
    rounded = torch.tensor(cap, dtype=dtype, device="cpu")
    if float(rounded) > cap:
        rounded = torch.nextafter(rounded, torch.zeros_like(rounded))
    return float(rounded)


def apply_parameter_update(
    state, output_chart, input_chart, p, displacement, *, step_size, amplitude_max=None
):
    if amplitude_max is None:
        amplitude_max = float(state.scalar("amplitude_max"))
    encoded = _rounded_amplitude_max(amplitude_max, p.dtype)
    proposal = p + displacement
    return torch.stack(
        (
            proposal[:, 0].clamp(-encoded, encoded),
            proposal[:, 1].clamp(0, 1),
            proposal[:, 2],
            proposal[:, 3],
        ),
        dim=1,
    )


def transport_parameter_state(state, output_chart, input_chart, old, new, vector):
    return vector
