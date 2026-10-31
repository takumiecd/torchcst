"""Physical radius proposal projection, distinct from the legacy sigma law."""

import torch


def project_parameter_gradient(state, input_chart, output_chart, p, gradient):
    return gradient


def apply_parameter_update(
    state, input_chart, output_chart, p, displacement, *, step_size
):
    proposal = p + displacement
    exact = (
        torch.minimum(
            input_chart.geometry.periods[0].double(),
            output_chart.geometry.periods[0].double(),
        )
        / 2
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


def transport_parameter_state(state, input_chart, output_chart, old, new, vector):
    return vector
