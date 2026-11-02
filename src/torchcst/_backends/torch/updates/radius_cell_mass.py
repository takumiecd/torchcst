"""Physical radius proposal projection, distinct from the legacy sigma law."""

import torch


def project_parameter_gradient(state, output_chart, input_chart, p, gradient):
    return gradient


def apply_parameter_update(
    state, output_chart, input_chart, p, displacement, *, step_size
):
    proposal = p + displacement
    exact = p.new_tensor(
        min(output_chart.geometry.periods[0], input_chart.geometry.periods[0]) / 2,
        dtype=torch.float64,
    )
    rounded = exact.to(p)
    ceiling = torch.where(
        rounded.double() > exact,
        torch.nextafter(rounded, torch.zeros_like(rounded)),
        rounded,
    )
    radius = torch.minimum(
        proposal[:, 1].clamp_min(state.scalar("radius_floor")), ceiling
    )
    return torch.stack((proposal[:, 0], radius, proposal[:, 2], proposal[:, 3]), dim=1)


def transport_parameter_state(state, output_chart, input_chart, old, new, vector):
    return vector
