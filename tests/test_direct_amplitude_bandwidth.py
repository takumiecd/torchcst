import pytest
import torch

from torchcst import (
    Chart,
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    ParameterAdamConfig,
    PolarAmpWidth,
)


def charts() -> tuple[Chart, Chart]:
    return Chart.linspace(3, low=-1.0, high=1.0), Chart.linspace(
        4, low=-1.0, high=1.0
    )


def kernel(**kwargs) -> DirectAmpWidth:
    options = {
        "amplitude_max": 2.0,
        "sigma_min": 0.1,
        "sigma_max": 1.0,
        "w_c": 0.5,
        "kappa": 3.0,
        "radial_regularization": 0.2,
    }
    options.update(kwargs)
    return DirectAmpWidth(**options)


def test_initialization_stores_direct_amplitude_and_q() -> None:
    torch.manual_seed(12)
    input_chart, output_chart = charts()
    value = kernel(alpha_init=0.03)
    atoms = 4096

    p = value.initialize(input_chart, output_chart, atoms, mode="uniform")

    assert p.shape == (atoms, 4)
    torch.testing.assert_close(p[:, 0], value.amplitude(input_chart, output_chart, p))
    torch.testing.assert_close(p[:, 1], torch.full_like(p[:, 1], 1.09))
    torch.testing.assert_close(
        value.bandwidth_alpha(input_chart, output_chart, p),
        torch.full_like(p[:, 1], 0.03),
    )
    assert p[:, 0].abs().max() <= value.amplitude_max
    assert abs(float(p[:, 0].std()) - 0.1 / atoms**0.5) < 0.05 * (
        0.1 / atoms**0.5
    )


def test_direct_and_polar_coordinates_materialize_the_same_atoms() -> None:
    input_chart, output_chart = charts()
    direct = kernel().to(dtype=torch.float64)
    polar = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
    ).to(dtype=torch.float64)
    w = torch.tensor([-1.2, 0.0, 1.4], dtype=torch.float64)
    q = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float64)
    center = torch.tensor(
        [[-0.7, -0.2], [0.1, 0.3], [0.8, 0.6]], dtype=torch.float64
    )
    direct_p = torch.cat((torch.stack((w, q), dim=-1), center), dim=-1)
    ratio = w / direct.amplitude_max
    polar_direction = torch.stack((ratio, (1.0 - ratio.square()).sqrt()), dim=-1)
    polar_p = torch.cat((polar_direction * q.sqrt().unsqueeze(-1), center), dim=-1)

    torch.testing.assert_close(
        direct.materialize_atoms(input_chart, output_chart, direct_p),
        polar.materialize_atoms(input_chart, output_chart, polar_p),
    )


def test_task_gradient_is_direct_w_gradient_and_q_has_no_gradient() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    p = torch.tensor([[0.6, 2.5, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)

    loss = value.materialize_atoms(input_chart, output_chart, p).sum()
    gradient = torch.autograd.grad(loss, p)[0]

    assert gradient[0, 0] != 0
    torch.testing.assert_close(gradient[:, 1], torch.zeros_like(gradient[:, 1]))


def test_update_uses_accepted_physical_amplitude_displacement() -> None:
    input_chart, output_chart = charts()
    value = kernel(radial_regularization=0.0).to(dtype=torch.float64)
    p = torch.tensor([[0.6, 1.5, 0.1, -0.2]], dtype=torch.float64)
    displacement = torch.tensor([[0.2, 9.0, 0.3, -0.1]], dtype=torch.float64)

    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        displacement,
        step_size=0.5,
    )

    accepted_delta = (updated[:, 0] - p[:, 0]) / value.amplitude_max
    expected_q = p[:, 1] + p[:, 1] * accepted_delta.square()
    torch.testing.assert_close(updated[:, 0], torch.tensor([0.8], dtype=p.dtype))
    torch.testing.assert_close(updated[:, 1], expected_q)
    torch.testing.assert_close(updated[:, 2:], p[:, 2:] + displacement[:, 2:])


def test_direct_rejects_polar_only_activity_controls() -> None:
    for option in (
        {"activity_gain": 2.0},
        {"activity_mode": "time_energy"},
        {"dormant_expansion_rate": 10.0},
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            kernel(**option)


def test_q_regularization_is_euclidean() -> None:
    input_chart, output_chart = charts()
    value = kernel(radial_regularization=0.3).to(dtype=torch.float64)
    p = torch.tensor([[0.2, 1.4, 0.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[0.1, -7.0, 0.0, 0.0]], dtype=torch.float64)
    step_size = 0.25

    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        displacement,
        step_size=step_size,
    )

    accepted_delta = (updated[:, 0] - p[:, 0]) / value.amplitude_max
    geometric_q = p[:, 1] + p[:, 1] * accepted_delta.square()
    expected_q = geometric_q - step_size * 0.3 * (geometric_q - 1.0)
    torch.testing.assert_close(updated[:, 1], expected_q)


def test_large_amplitude_proposal_stays_in_finite_chord_domain() -> None:
    input_chart, output_chart = charts()
    value = kernel(radial_regularization=0.0).to(dtype=torch.float64)
    p = torch.tensor([[-2.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        displacement,
        step_size=0.5,
    )

    assert torch.isfinite(updated).all()
    assert torch.all(updated[:, 0].abs() <= value.amplitude_max)
    torch.testing.assert_close(updated[:, 1], torch.tensor([4.0], dtype=p.dtype))


def test_parameter_adam_moments_are_direct_w_moments() -> None:
    input_chart, output_chart = charts()
    value = kernel(radial_regularization=0.0)
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=1,
        kernel=value,
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTParameterAdam(
        model,
        cst=ParameterAdamConfig(
            lr=0.01,
            betas=(0.5, 0.9),
            decay_steps=None,
        ),
    )

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, input_chart.features, dtype=torch.float64)).sum()
    loss.backward()
    direct_gradient = model.atoms.p.grad.detach().clone()
    optimizer.step()

    state = optimizer.state[model.atoms.p]
    torch.testing.assert_close(
        state["exp_avg"][:, 0],
        0.5 * direct_gradient[:, 0],
    )
    torch.testing.assert_close(
        state["exp_avg_sq"][:, 0],
        0.1 * direct_gradient[:, 0].square(),
    )
    torch.testing.assert_close(
        state["exp_avg"][:, 1], torch.zeros_like(state["exp_avg"][:, 1])
    )
    torch.testing.assert_close(
        state["exp_avg_sq"][:, 1], torch.zeros_like(state["exp_avg_sq"][:, 1])
    )
