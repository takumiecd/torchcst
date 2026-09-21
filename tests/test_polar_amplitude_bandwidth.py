import pytest
import torch

from torchcst import (
    Chart,
    CSTLinear,
    CSTNormalizedSGD,
    CSTParameterAdam,
    NormalizedOptimizerConfig,
    ParameterAdamConfig,
    PolarAmpWidth,
)


def charts() -> tuple[Chart, Chart]:
    return Chart.linspace(3, low=-1.0, high=1.0), Chart.linspace(4, low=-1.0, high=1.0)


def kernel() -> PolarAmpWidth:
    return PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
    )


def test_initialization_uses_unit_radius_and_small_bounded_amplitudes() -> None:
    torch.manual_seed(12)
    input_chart, output_chart = charts()
    value = kernel()
    atoms = 4096
    p = value.initialize(input_chart, output_chart, atoms, mode="uniform")

    radius_square = p[:, :2].square().sum(dim=-1)
    amplitude = value.amplitude(input_chart, output_chart, p)
    alpha = value.bandwidth_alpha(input_chart, output_chart, p)

    assert p.shape == (atoms, 4)
    torch.testing.assert_close(radius_square, torch.ones_like(radius_square))
    torch.testing.assert_close(alpha, torch.zeros_like(alpha), atol=1e-6, rtol=0)
    assert amplitude.abs().max() < value.amplitude_max
    assert abs(float(amplitude.std()) - 0.1 / atoms**0.5) < 0.05 * (0.1 / atoms**0.5)


def test_alpha_initialization_preserves_amplitude_and_sets_radius() -> None:
    torch.manual_seed(12)
    input_chart, output_chart = charts()
    baseline = kernel()
    initialized = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        kappa=3.0,
        alpha_init=0.03,
        radial_regularization=0.2,
    )
    p_baseline = baseline.initialize(input_chart, output_chart, 32, mode="uniform")
    torch.manual_seed(12)
    p_initialized = initialized.initialize(
        input_chart, output_chart, 32, mode="uniform"
    )

    torch.testing.assert_close(
        initialized.amplitude(input_chart, output_chart, p_initialized),
        baseline.amplitude(input_chart, output_chart, p_baseline),
    )
    torch.testing.assert_close(
        initialized.bandwidth_alpha(input_chart, output_chart, p_initialized),
        torch.full((32,), 0.03),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        p_initialized[:, :2].square().sum(dim=-1),
        torch.full((32,), 1.09),
    )


def test_polar_map_decouples_angular_amplitude_and_radial_alpha() -> None:
    value = kernel()
    polar = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)

    amplitude, alpha = value._amplitude_and_alpha(polar)
    amplitude_gradient = torch.autograd.grad(amplitude.sum(), polar, retain_graph=True)[
        0
    ]
    alpha_gradient = torch.autograd.grad(alpha.sum(), polar)[0]
    radial = polar.detach()
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=-1)

    torch.testing.assert_close(
        (amplitude_gradient * radial).sum(),
        torch.tensor(0.0, dtype=polar.dtype),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(
        (alpha_gradient * tangent).sum(),
        torch.tensor(0.0, dtype=polar.dtype),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(amplitude, torch.tensor([1.2], dtype=polar.dtype))
    torch.testing.assert_close(alpha, torch.tensor([0.0], dtype=polar.dtype))


def test_bandwidth_interpolates_between_rational_bounds() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    # w = W / 4 = w_c and q = 1, 2.5, 4 respectively.
    radii = torch.tensor([1.0, 2.5**0.5, 2.0], dtype=torch.float64)
    sine = torch.tensor(0.25, dtype=torch.float64)
    cosine = (1 - sine.square()).sqrt()
    polar = radii[:, None] * torch.stack((sine, cosine))[None]
    p = torch.cat((polar, torch.zeros(3, 2, dtype=torch.float64)), dim=-1)

    lower, upper = value.bandwidth_bounds(input_chart, output_chart, p)
    sigma = value.bandwidth_sigma(input_chart, output_chart, p)
    alpha = value.bandwidth_alpha(input_chart, output_chart, p)

    torch.testing.assert_close(
        alpha, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(lower, torch.full_like(lower, 0.325))
    torch.testing.assert_close(upper, torch.full_like(upper, 0.775))
    torch.testing.assert_close(
        sigma, torch.tensor([0.325, 0.55, 0.775], dtype=torch.float64)
    )


def test_lower_kappa_widens_only_lower_bandwidth_bound() -> None:
    input_chart, output_chart = charts()
    baseline = kernel().to(dtype=torch.float64)
    widened = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        kappa=3.0,
        lower_kappa=1.0,
        radial_regularization=0.2,
    ).to(dtype=torch.float64)
    polar = torch.tensor([[0.25, 3**0.5 / 2]], dtype=torch.float64)
    p = torch.cat((polar, torch.zeros(1, 2, dtype=torch.float64)), dim=-1)

    baseline_lower, baseline_upper = baseline.bandwidth_bounds(
        input_chart, output_chart, p
    )
    widened_lower, widened_upper = widened.bandwidth_bounds(
        input_chart, output_chart, p
    )

    assert torch.all(widened_lower > baseline_lower)
    torch.testing.assert_close(widened_upper, baseline_upper)


def test_upper_floor_preserves_alpha_authority_at_high_amplitude() -> None:
    input_chart, output_chart = charts()
    value = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.01,
        kappa=3.0,
        upper_floor=0.2,
        radial_regularization=0.2,
    ).to(dtype=torch.float64)
    # Maximum angular amplitude with alpha = 0, 0.5, and 1.
    radii = torch.tensor([1.0, 2.5**0.5, 2.0], dtype=torch.float64)
    polar = torch.stack((radii, torch.zeros_like(radii)), dim=-1)
    p = torch.cat((polar, torch.zeros(3, 2, dtype=torch.float64)), dim=-1)

    lower, upper = value.bandwidth_bounds(input_chart, output_chart, p)
    sigma = value.bandwidth_sigma(input_chart, output_chart, p)

    torch.testing.assert_close(upper, torch.full_like(upper, 0.2))
    torch.testing.assert_close(sigma[0], lower[0])
    torch.testing.assert_close(sigma[1], (lower[1] + upper[1]) / 2)
    torch.testing.assert_close(sigma[2], upper[2])


def test_alpha_clamp_keeps_sigma_inside_bounds_for_arbitrary_radius() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    polar = torch.tensor(
        [[0.1, 0.2], [0.5, 0.5], [2.0, 2.0], [20.0, 20.0]],
        dtype=torch.float64,
    )
    p = torch.cat((polar, torch.zeros(4, 2, dtype=torch.float64)), dim=-1)
    lower, upper = value.bandwidth_bounds(input_chart, output_chart, p)
    sigma = value.bandwidth_sigma(input_chart, output_chart, p)
    alpha = value.bandwidth_alpha(input_chart, output_chart, p)

    assert torch.all((0 <= alpha) & (alpha <= 1))
    assert torch.all(value.sigma_min <= lower)
    assert torch.all(lower <= sigma)
    assert torch.all(sigma <= upper)
    assert torch.all(upper <= value.sigma_max)


def test_factorization_and_second_derivatives_are_finite() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    # Interior alpha avoids testing the deliberately nonsmooth clamp boundary.
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.1, -0.2]], dtype=torch.float64)
    p[:, :2] *= radius

    represented = value.materialize_atoms(input_chart, output_chart, p)
    phi_input, phi_output = value.factors(input_chart, output_chart, p)
    hessian = torch.func.hessian(
        lambda atom: value.materialize_atoms(
            input_chart, output_chart, atom.unsqueeze(0)
        ).sum()
    )(p[0])

    torch.testing.assert_close(
        represented, torch.einsum("oa,ia->aoi", phi_output, phi_input)
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        value.amplitude(input_chart, output_chart, p).abs(),
    )
    assert torch.isfinite(hessian).all()


def test_task_gradient_cannot_learn_bandwidth_radially() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        p[:, :2] *= radius

    loss = value.materialize_atoms(input_chart, output_chart, p).sum()
    polar_gradient = torch.autograd.grad(loss, p)[0][:, :2]

    torch.testing.assert_close(
        (polar_gradient * p[:, :2]).sum(),
        torch.tensor(0.0, dtype=p.dtype),
        atol=1e-13,
        rtol=0,
    )


def test_parameter_update_projects_task_motion_and_regularizes_radius() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    # The inward component is discarded; only the horizontal tangent remains.
    displacement = torch.tensor([[0.3, -0.8, 0.2, -0.1]], dtype=torch.float64)

    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        displacement,
        step_size=0.5,
    )
    q = updated[:, :2].square().sum(dim=-1)
    task_q = torch.tensor([1.09], dtype=p.dtype)
    decay = torch.exp(torch.tensor(-4 * 0.2 * 0.5, dtype=p.dtype))
    expected_q = 1 / (1 - ((task_q - 1) / task_q) * decay)

    torch.testing.assert_close(q, expected_q)
    assert torch.all((1 <= q) & (q <= 4))
    torch.testing.assert_close(updated[:, 2:], p[:, 2:] + displacement[:, 2:])


def test_activity_gain_changes_radius_without_changing_amplitude() -> None:
    input_chart, output_chart = charts()
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    displacement = torch.tensor([[0.3, -0.8, 0.2, -0.1]], dtype=torch.float64)
    results = []
    for gain in (1.0, 10.0):
        value = PolarAmpWidth(
            amplitude_max=2.0,
            sigma_min=0.1,
            sigma_max=1.0,
            w_c=0.5,
            activity_gain=gain,
            radial_regularization=0.0,
        ).to(dtype=torch.float64)
        updated = value.apply_parameter_update(
            input_chart,
            output_chart,
            p,
            displacement,
            step_size=0.5,
        )
        results.append((value, updated))

    amplitudes = [
        value.amplitude(input_chart, output_chart, updated)
        for value, updated in results
    ]
    q = [updated[:, :2].square().sum(dim=-1) for _, updated in results]
    torch.testing.assert_close(amplitudes[0], amplitudes[1])
    torch.testing.assert_close(q[0], torch.tensor([1.09], dtype=p.dtype))
    torch.testing.assert_close(q[1], torch.tensor([1.9], dtype=p.dtype))


def test_time_energy_activity_uses_outer_step_size() -> None:
    input_chart, output_chart = charts()
    value = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        activity_gain=27.0,
        activity_mode="time_energy",
        radial_regularization=0.0,
    ).to(dtype=torch.float64)
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    displacement = torch.tensor([[0.01, -0.8, 0.0, 0.0]], dtype=torch.float64)

    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        displacement,
        step_size=0.002,
    )

    q = updated[:, :2].square().sum(dim=-1)
    torch.testing.assert_close(q, torch.tensor([2.35], dtype=p.dtype))


def test_radial_regularizer_preserves_amplitude_and_converges_to_unit_radius() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.0, 0.0]], dtype=torch.float64)
    p[:, :2] *= radius
    before = value.amplitude(input_chart, output_chart, p)

    for _ in range(100):
        p = value.apply_parameter_update(
            input_chart,
            output_chart,
            p,
            torch.zeros_like(p),
            step_size=0.5,
        )

    after = value.amplitude(input_chart, output_chart, p)
    q = p[:, :2].square().sum(dim=-1)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(q, torch.ones_like(q), atol=1e-12, rtol=0)


def test_parameter_update_projects_arbitrary_points_onto_annulus() -> None:
    input_chart, output_chart = charts()
    value = PolarAmpWidth(
        amplitude_max=2.0,
        sigma_min=0.1,
        sigma_max=1.0,
        w_c=0.5,
        radial_regularization=0.0,
    ).to(dtype=torch.float64)
    p = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 10.0, 0.0, 0.0]], dtype=torch.float64)
    updated = value.apply_parameter_update(
        input_chart,
        output_chart,
        p,
        torch.zeros_like(p),
        step_size=0.1,
    )
    q = updated[:, :2].square().sum(dim=-1)
    torch.testing.assert_close(q, torch.tensor([1.0, 4.0], dtype=p.dtype))


def test_parameter_adam_uses_kernel_update_geometry() -> None:
    input_chart, output_chart = charts()
    value = kernel()
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
            lr=0.1,
            betas=(0.0, 0.0),
            decay_steps=None,
        ),
    )
    with torch.no_grad():
        model.atoms.p[0, :2] = torch.tensor([0.6, 0.8]) * 2.5**0.5
    before = value.amplitude(input_chart, output_chart, model.atoms.p).detach()
    before_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.zeros(1, input_chart.features, dtype=torch.float64)).sum() * 0
    loss.backward()
    optimizer.step()

    after = value.amplitude(input_chart, output_chart, model.atoms.p).detach()
    after_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()
    torch.testing.assert_close(after, before)
    assert torch.all(after_q < before_q)
    assert torch.all(after_q >= 1)


def test_model_optimizer_uses_kernel_update_geometry() -> None:
    input_chart, output_chart = charts()
    value = kernel()
    model = CSTLinear(
        input_chart,
        output_chart,
        atoms=1,
        kernel=value,
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTNormalizedSGD(
        model,
        cst=NormalizedOptimizerConfig(
            lr=0.1,
            trust_radius=0.2,
            initial_zero_step=True,
            kernel_step_size=0.5,
        ),
        dense=None,
    )
    with torch.no_grad():
        model.atoms.p[0, :2] = torch.tensor([0.6, 0.8]) * 2.5**0.5
    before = value.amplitude(input_chart, output_chart, model.atoms.p).detach()
    before_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.zeros(1, input_chart.features, dtype=torch.float64)).sum() * 0
    loss.backward()
    optimizer.step()

    after = value.amplitude(input_chart, output_chart, model.atoms.p).detach()
    after_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()
    decay = torch.exp(torch.tensor(-4 * 0.2 * 0.5, dtype=after_q.dtype))
    expected_q = 1 / (1 - ((before_q - 1) / before_q) * decay)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(after_q, expected_q)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"amplitude_max": 0.0}, "amplitude_max"),
        ({"w_c": 0.0}, "w_c"),
        ({"kappa": 1.0}, "kappa"),
        ({"lower_kappa": 0.0}, "lower_kappa"),
        ({"activity_gain": 0.0}, "activity_gain"),
        ({"alpha_init": -0.1}, "alpha_init"),
        ({"alpha_init": 1.1}, "alpha_init"),
        ({"activity_mode": "unknown"}, "activity_mode"),
        ({"radial_regularization": -1.0}, "radial_regularization"),
        ({"sigma_min": 2.0, "sigma_max": 1.0}, "sigma_max"),
    ],
)
def test_configuration_validation(kwargs: dict[str, object], message: str) -> None:
    options = {
        "amplitude_max": 2.0,
        "sigma_min": 0.1,
        "sigma_max": 1.0,
        "w_c": 0.5,
        "kappa": 3.0,
        "radial_regularization": 0.2,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PolarAmpWidth(**options)
