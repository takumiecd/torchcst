"""Projected a/q proposals; Adam moments retain their ordinary meaning."""

import torch


def project_parameter_gradient(state, output_chart, input_chart, p, gradient):
    return gradient


def apply_parameter_update(
    state, output_chart, input_chart, p, displacement, *, step_size
):
    proposal = p + displacement
    cap = state.scalar("amplitude_max")
    encoded = cap.to(p)
    encoded = torch.where(
        encoded.double() > cap,
        torch.nextafter(encoded, torch.zeros_like(encoded)),
        encoded,
    )
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
