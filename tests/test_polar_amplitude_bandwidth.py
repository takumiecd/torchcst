from __future__ import annotations

import pytest
import torch
from kernel_cases import polar_state

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst.geometry.state import ChartState


def charts() -> tuple[ChartState, ChartState]:
    return (
        _construction.linspace(3, low=-1.0, high=1.0),
        _construction.linspace(4, low=-1.0, high=1.0),
    )


def kernel() -> polar_state:
    return polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    )


def test_initialization_uses_unit_radius_and_small_bounded_amplitudes() -> None:
    torch.manual_seed(12)
    input_chart, output_chart = charts()
    value = kernel()
    atoms = 4096
    p = _kernel.initialize(value, input_chart, output_chart, atoms, mode="uniform")
    radius_square = p[:, :2].square().sum(dim=-1)
    amplitude = _kernel.coordinate(value, "amplitude", input_chart, output_chart, p)
    alpha = _kernel.coordinate(value, "bandwidth_alpha", input_chart, output_chart, p)
    assert p.shape == (atoms, 4)
    torch.testing.assert_close(radius_square, torch.ones_like(radius_square))
    torch.testing.assert_close(alpha, torch.zeros_like(alpha), atol=1e-06, rtol=0)
    assert amplitude.abs().max() < value.scalar("amplitude_max")
    assert abs(float(amplitude.std()) - 0.1 / atoms**0.5) < 0.05 * (0.1 / atoms**0.5)


def test_alpha_initialization_preserves_amplitude_and_sets_radius() -> None:
    torch.manual_seed(12)
    input_chart, output_chart = charts()
    baseline = kernel()
    initialized = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        alpha_init=0.03,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    )
    p_baseline = _kernel.initialize(
        baseline, input_chart, output_chart, 32, mode="uniform"
    )
    torch.manual_seed(12)
    p_initialized = _kernel.initialize(
        initialized, input_chart, output_chart, 32, mode="uniform"
    )
    torch.testing.assert_close(
        _kernel.coordinate(
            initialized, "amplitude", input_chart, output_chart, p_initialized
        ),
        _kernel.coordinate(
            baseline, "amplitude", input_chart, output_chart, p_baseline
        ),
    )
    torch.testing.assert_close(
        _kernel.coordinate(
            initialized, "bandwidth_alpha", input_chart, output_chart, p_initialized
        ),
        torch.full((32,), 0.03),
        atol=1e-06,
        rtol=0,
    )
    torch.testing.assert_close(
        p_initialized[:, :2].square().sum(dim=-1), torch.full((32,), 1.09)
    )


def test_polar_map_decouples_angular_amplitude_and_radial_alpha() -> None:
    value = kernel()
    polar = torch.tensor([[0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    amplitude, alpha = _kernel.coordinate(value, "_amplitude_and_alpha", polar)
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
    radii = torch.tensor([1.0, 2.5**0.5, 2.0], dtype=torch.float64)
    sine = torch.tensor(0.25, dtype=torch.float64)
    cosine = (1 - sine.square()).sqrt()
    polar = radii[:, None] * torch.stack((sine, cosine))[None]
    p = torch.cat((polar, torch.zeros(3, 2, dtype=torch.float64)), dim=-1)
    lower, upper = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    sigma = _kernel.coordinate(value, "bandwidth_sigma", input_chart, output_chart, p)
    alpha = _kernel.coordinate(value, "bandwidth_alpha", input_chart, output_chart, p)
    torch.testing.assert_close(
        alpha, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(lower, torch.full_like(lower, 0.325))
    torch.testing.assert_close(upper, torch.full_like(upper, 0.775))
    torch.testing.assert_close(
        sigma, torch.tensor([0.325, (0.325 * 0.775) ** 0.5, 0.775], dtype=torch.float64)
    )


def test_birth_width_separates_zero_amplitude_lower_and_upper_bounds() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=0.25, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    radii = torch.tensor([1.0, 2.5**0.5, 2.0], dtype=torch.float64)
    polar = torch.stack((torch.zeros_like(radii), radii), dim=-1)
    p = torch.cat((polar, torch.zeros(3, 2, dtype=torch.float64)), dim=-1)
    lower, upper = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    sigma = _kernel.coordinate(value, "bandwidth_sigma", input_chart, output_chart, p)
    torch.testing.assert_close(lower, torch.full_like(lower, 0.25))
    torch.testing.assert_close(upper, torch.ones_like(upper))
    torch.testing.assert_close(
        sigma, torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
    )


def test_input_and_output_bandwidths_can_be_configured_independently() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
        profile=None,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=0.8, birth=0.2, upper_floor=0.1
        ),
        output_bounds=BandwidthBounds(
            minimum=0.15, maximum=1.2, birth=0.3, upper_floor=0.15
        ),
    ).to(dtype=torch.float64)
    polar = torch.tensor([[0.0, 2.5**0.5]], dtype=torch.float64)
    p = torch.cat((polar, torch.zeros(1, 2, dtype=torch.float64)), dim=-1)
    sigma_input, sigma_output = _kernel.coordinate(
        value, "bandwidth_sigmas", input_chart, output_chart, p
    )
    input_bounds, output_bounds = _kernel.coordinate(
        value, "bandwidth_bounds_by_side", input_chart, output_chart, p
    )
    torch.testing.assert_close(sigma_input, torch.tensor([0.4], dtype=torch.float64))
    torch.testing.assert_close(sigma_output, torch.tensor([0.6], dtype=torch.float64))
    torch.testing.assert_close(
        input_bounds[0], torch.tensor([0.2], dtype=torch.float64)
    )
    torch.testing.assert_close(
        input_bounds[1], torch.tensor([0.8], dtype=torch.float64)
    )
    torch.testing.assert_close(
        output_bounds[0], torch.tensor([0.3], dtype=torch.float64)
    )
    torch.testing.assert_close(
        output_bounds[1], torch.tensor([1.2], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="by-side"):
        _kernel.coordinate(value, "bandwidth_sigma", input_chart, output_chart, p)


def test_lower_kappa_widens_only_lower_bandwidth_bound() -> None:
    input_chart, output_chart = charts()
    baseline = kernel().to(dtype=torch.float64)
    widened = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        lower_kappa=1.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    polar = torch.tensor([[0.25, 3**0.5 / 2]], dtype=torch.float64)
    p = torch.cat((polar, torch.zeros(1, 2, dtype=torch.float64)), dim=-1)
    baseline_lower, baseline_upper = _kernel.coordinate(
        baseline, "bandwidth_bounds", input_chart, output_chart, p
    )
    widened_lower, widened_upper = _kernel.coordinate(
        widened, "bandwidth_bounds", input_chart, output_chart, p
    )
    assert torch.all(widened_lower > baseline_lower)
    torch.testing.assert_close(widened_upper, baseline_upper)


def test_lower_half_amplitude_directly_sets_lower_midpoint() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
        lower_kappa=(0.5 / 0.25) ** 2,
    ).to(dtype=torch.float64)
    polar = torch.tensor([[0.125, (1 - 0.125**2) ** 0.5]], dtype=torch.float64)
    p = torch.cat((polar, torch.zeros(1, 2, dtype=torch.float64)), dim=-1)
    lower, _ = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    torch.testing.assert_close(lower, torch.tensor([0.55], dtype=torch.float64))
    torch.testing.assert_close(
        _kernel.coordinate(value, "lower_half_amplitude"),
        torch.tensor(0.25, dtype=torch.float64),
    )


def test_lower_half_amplitude_preserves_legacy_lower_curve() -> None:
    input_chart, output_chart = charts()
    legacy = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        lower_kappa=12.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    direct = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
        lower_kappa=(0.5 / (0.5 / 12**0.5)) ** 2,
    ).to(dtype=torch.float64)
    amplitudes = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float64)
    polar = torch.stack(
        (amplitudes / 2.0, (1 - (amplitudes / 2.0).square()).sqrt()), dim=-1
    )
    p = torch.cat((polar, torch.zeros(4, 2, dtype=torch.float64)), dim=-1)
    legacy_lower, legacy_upper = _kernel.coordinate(
        legacy, "bandwidth_bounds", input_chart, output_chart, p
    )
    direct_lower, direct_upper = _kernel.coordinate(
        direct, "bandwidth_bounds", input_chart, output_chart, p
    )
    torch.testing.assert_close(direct_lower, legacy_lower)
    torch.testing.assert_close(direct_upper, legacy_upper)


def test_upper_floor_preserves_alpha_authority_at_high_amplitude() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.01,
        kappa=3.0,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.2
        ),
    ).to(dtype=torch.float64)
    radii = torch.tensor([1.0, 2.5**0.5, 2.0], dtype=torch.float64)
    polar = torch.stack((radii, torch.zeros_like(radii)), dim=-1)
    p = torch.cat((polar, torch.zeros(3, 2, dtype=torch.float64)), dim=-1)
    lower, upper = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    sigma = _kernel.coordinate(value, "bandwidth_sigma", input_chart, output_chart, p)
    torch.testing.assert_close(upper, torch.full_like(upper, 0.2))
    torch.testing.assert_close(sigma[0], lower[0])
    torch.testing.assert_close(sigma[1], (lower[1] * upper[1]).sqrt())
    torch.testing.assert_close(sigma[2], upper[2])


def test_smaller_upper_decay_power_delays_only_upper_collapse() -> None:
    input_chart, output_chart = charts()
    baseline = kernel().to(dtype=torch.float64)
    delayed = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        kappa=3.0,
        upper_decay_power=0.75,
        radial_regularization=0.2,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    polar = torch.tensor([[0.75, (1 - 0.75**2) ** 0.5]], dtype=torch.float64)
    p = torch.cat((polar, torch.zeros(1, 2, dtype=torch.float64)), dim=-1)
    baseline_lower, baseline_upper = _kernel.coordinate(
        baseline, "bandwidth_bounds", input_chart, output_chart, p
    )
    delayed_lower, delayed_upper = _kernel.coordinate(
        delayed, "bandwidth_bounds", input_chart, output_chart, p
    )
    torch.testing.assert_close(delayed_lower, baseline_lower)
    assert torch.all(delayed_upper > baseline_upper)


def test_alpha_clamp_keeps_sigma_inside_bounds_for_arbitrary_radius() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    polar = torch.tensor(
        [[0.1, 0.2], [0.5, 0.5], [2.0, 2.0], [20.0, 20.0]], dtype=torch.float64
    )
    p = torch.cat((polar, torch.zeros(4, 2, dtype=torch.float64)), dim=-1)
    lower, upper = _kernel.coordinate(
        value, "bandwidth_bounds", input_chart, output_chart, p
    )
    sigma = _kernel.coordinate(value, "bandwidth_sigma", input_chart, output_chart, p)
    alpha = _kernel.coordinate(value, "bandwidth_alpha", input_chart, output_chart, p)
    assert torch.all((0 <= alpha) & (alpha <= 1))
    assert torch.all(value.scalar("sigma_min_input") <= lower)
    assert torch.all(lower <= sigma)
    assert torch.all(sigma <= upper)
    assert torch.all(upper <= value.scalar("sigma_max_input"))


def test_factorization_and_second_derivatives_are_finite() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.1, -0.2]], dtype=torch.float64)
    p[:, :2] *= radius
    represented = _kernel.materialize_atoms(value, input_chart, output_chart, p)
    phi_input, phi_output = _kernel.factors(value, input_chart, output_chart, p)
    hessian = torch.func.hessian(
        lambda atom: _kernel.materialize_atoms(
            value, input_chart, output_chart, atom.unsqueeze(0)
        ).sum()
    )(p[0])
    torch.testing.assert_close(
        represented, torch.einsum("oa,ia->aoi", phi_output, phi_input)
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        _kernel.coordinate(value, "amplitude", input_chart, output_chart, p).abs(),
    )
    assert torch.isfinite(hessian).all()


def test_task_gradient_cannot_learn_bandwidth_radially() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        p[:, :2] *= radius
    loss = _kernel.materialize_atoms(value, input_chart, output_chart, p).sum()
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
    displacement = torch.tensor([[0.3, -0.8, 0.2, -0.1]], dtype=torch.float64)
    updated = _kernel.apply_parameter_update(
        value, input_chart, output_chart, p, displacement, step_size=0.5
    )
    q = updated[:, :2].square().sum(dim=-1)
    task_q = torch.tensor([1.09], dtype=p.dtype)
    decay = torch.exp(torch.tensor(-4 * 0.2 * 0.5, dtype=p.dtype))
    expected_q = 1 / (1 - (task_q - 1) / task_q * decay)
    torch.testing.assert_close(q, expected_q)
    assert torch.all((1 <= q) & (q <= 4))
    torch.testing.assert_close(updated[:, 2:], p[:, 2:] + displacement[:, 2:])


def test_activity_gain_changes_radius_without_changing_amplitude() -> None:
    input_chart, output_chart = charts()
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    displacement = torch.tensor([[0.3, -0.8, 0.2, -0.1]], dtype=torch.float64)
    results = []
    for gain in (1.0, 10.0):
        value = polar_state(
            amplitude_max=2.0,
            w_c=0.5,
            activity_gain=gain,
            radial_regularization=0.0,
            input_bounds=BandwidthBounds(
                minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
            ),
        ).to(dtype=torch.float64)
        updated = _kernel.apply_parameter_update(
            value, input_chart, output_chart, p, displacement, step_size=0.5
        )
        results.append((value, updated))
    amplitudes = [
        _kernel.coordinate(value, "amplitude", input_chart, output_chart, updated)
        for value, updated in results
    ]
    q = [updated[:, :2].square().sum(dim=-1) for _, updated in results]
    torch.testing.assert_close(amplitudes[0], amplitudes[1])
    torch.testing.assert_close(q[0], torch.tensor([1.09], dtype=p.dtype))
    torch.testing.assert_close(q[1], torch.tensor([1.9], dtype=p.dtype))


def test_time_energy_activity_uses_outer_step_size() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        activity_gain=27.0,
        activity_mode="time_energy",
        radial_regularization=0.0,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    displacement = torch.tensor([[0.01, -0.8, 0.0, 0.0]], dtype=torch.float64)
    updated = _kernel.apply_parameter_update(
        value, input_chart, output_chart, p, displacement, step_size=0.002
    )
    q = updated[:, :2].square().sum(dim=-1)
    torch.testing.assert_close(q, torch.tensor([2.35], dtype=p.dtype))


def test_dormant_expansion_advances_alpha_without_task_motion() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        dormant_expansion_rate=1.0,
        radial_regularization=0.0,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    p = torch.tensor([[0.0, 1.0, 0.1, -0.2]], dtype=torch.float64)
    updated = _kernel.apply_parameter_update(
        value, input_chart, output_chart, p, torch.zeros_like(p), step_size=0.5
    )
    torch.testing.assert_close(
        _kernel.coordinate(value, "amplitude", input_chart, output_chart, updated),
        torch.zeros(1, dtype=p.dtype),
    )
    torch.testing.assert_close(
        _kernel.coordinate(
            value, "bandwidth_alpha", input_chart, output_chart, updated
        ),
        torch.tensor([0.5], dtype=p.dtype),
    )
    torch.testing.assert_close(updated[:, 2:], p[:, 2:])


def test_dormant_expansion_is_suppressed_by_large_amplitude() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        dormant_expansion_rate=1.0,
        radial_regularization=0.0,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    p = torch.tensor([[1.0, 0.0, 0.1, -0.2]], dtype=torch.float64)
    updated = _kernel.apply_parameter_update(
        value, input_chart, output_chart, p, torch.zeros_like(p), step_size=0.5
    )
    expected_alpha = torch.tensor([0.5 / 17.0], dtype=p.dtype)
    torch.testing.assert_close(
        _kernel.coordinate(
            value, "bandwidth_alpha", input_chart, output_chart, updated
        ),
        expected_alpha,
    )


def test_radial_regularizer_preserves_amplitude_and_converges_to_unit_radius() -> None:
    input_chart, output_chart = charts()
    value = kernel().to(dtype=torch.float64)
    radius = torch.tensor(2.5**0.5, dtype=torch.float64)
    p = torch.tensor([[0.6, 0.8, 0.0, 0.0]], dtype=torch.float64)
    p[:, :2] *= radius
    before = _kernel.coordinate(value, "amplitude", input_chart, output_chart, p)
    for _ in range(100):
        p = _kernel.apply_parameter_update(
            value, input_chart, output_chart, p, torch.zeros_like(p), step_size=0.5
        )
    after = _kernel.coordinate(value, "amplitude", input_chart, output_chart, p)
    q = p[:, :2].square().sum(dim=-1)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(q, torch.ones_like(q), atol=1e-12, rtol=0)


def test_parameter_update_projects_arbitrary_points_onto_annulus() -> None:
    input_chart, output_chart = charts()
    value = polar_state(
        amplitude_max=2.0,
        w_c=0.5,
        radial_regularization=0.0,
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.0, birth=1.0, upper_floor=0.1
        ),
    ).to(dtype=torch.float64)
    p = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 10.0, 0.0, 0.0]], dtype=torch.float64)
    updated = _kernel.apply_parameter_update(
        value, input_chart, output_chart, p, torch.zeros_like(p), step_size=0.1
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
        kernel=value.declaration(),
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.1,
            betas=(0.0, 0.0),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    with torch.no_grad():
        model.atoms.p[0, :2] = torch.tensor([0.6, 0.8]) * 2.5**0.5
    before = _kernel.coordinate(
        value, "amplitude", input_chart, output_chart, model.atoms.p
    ).detach()
    before_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.zeros(1, input_chart.features, dtype=torch.float64)).sum() * 0
    loss.backward()
    optimizer.step()
    after = _kernel.coordinate(
        value, "amplitude", input_chart, output_chart, model.atoms.p
    ).detach()
    after_q = model.atoms.p[:, :2].square().sum(dim=-1).detach()
    torch.testing.assert_close(after, before)
    assert torch.all(after_q < before_q)
    assert torch.all(after_q >= 1)
