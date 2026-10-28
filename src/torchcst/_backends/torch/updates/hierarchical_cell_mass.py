"""Physical-coordinate proposals with explicitly declared sigma projection."""

import torch


def project_parameter_gradient(state, input_chart, output_chart, p, gradient):
    return gradient


def apply_parameter_update(
    state, input_chart, output_chart, p, displacement, *, step_size
):
    proposed = p + displacement
    # Evaluation never clips sigma. Only this declared optimizer policy projects
    # the proposal; amplitude and both unwrapped periodic centers stay Euclidean.
    exact = (
        torch.minimum(
            input_chart.geometry.periods[0].double(),
            output_chart.geometry.periods[0].double(),
        )
        / 3
    )
    rounded = exact.to(p)
    # Do not round the upper endpoint outside the mathematical domain. This
    # also keeps a subsequently double()-converted checkpoint valid.
    ceiling = torch.where(
        rounded.double() > exact,
        torch.nextafter(rounded, torch.zeros_like(rounded)),
        rounded,
    )
    sigma = torch.minimum(
        proposed[:, 1].clamp_min(state.scalar("sigma_floor")), ceiling
    )
    return torch.stack((proposed[:, 0], sigma, proposed[:, 2], proposed[:, 3]), dim=1)


def transport_parameter_state(state, input_chart, output_chart, old, new, vector):
    return vector
