import pytest
import torch

from torchcst import (
    Amplitude,
    AmplitudeBandwidthSeparable,
    Chart,
    Gaussian,
    Separable,
)


def charts() -> tuple[Chart, Chart]:
    return Chart.linspace(3, low=-1.0, high=1.0), Chart.grid((2, 2), low=-1.0, high=1.0)


def base_kernel() -> Separable:
    return Separable(
        input_profile=Gaussian(0.4),
        output_profile=Gaussian(0.7),
    )


def test_separable_kernel_is_the_amplitude_free_profile_product() -> None:
    input_chart, output_chart = charts()
    kernel = base_kernel()
    p = kernel.initialize(input_chart, output_chart, 2, mode="balanced")

    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)
    represented = kernel.materialize_atoms(input_chart, output_chart, p)

    assert kernel.parameter_dim(input_chart, output_chart) == 3
    assert p.shape == (2, 3)
    assert phi_input.shape == (3, 2)
    assert phi_output.shape == (4, 2)
    assert represented.shape == (2, 4, 3)
    torch.testing.assert_close(
        represented,
        torch.einsum("oa,ia->aoi", phi_output, phi_input),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        torch.ones(p.shape[0]),
    )


def test_amplitude_wrapper_adds_one_signed_coordinate_to_any_kernel() -> None:
    input_chart, output_chart = charts()
    base = base_kernel()
    kernel = Amplitude(base)
    p = kernel.initialize(input_chart, output_chart, 2, mode="balanced")

    represented = kernel.materialize_atoms(input_chart, output_chart, p)
    expected = p[:, 0, None, None] * base.materialize_atoms(
        input_chart,
        output_chart,
        p[:, 1:],
    )
    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)

    assert kernel.parameter_dim(input_chart, output_chart) == 4
    assert p.shape == (2, 4)
    torch.testing.assert_close(represented, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        p[:, 0].abs(),
    )
    torch.testing.assert_close(
        represented,
        torch.einsum("oa,ia->aoi", phi_output, phi_input),
    )


def test_amplitude_kernels_use_the_successful_small_weight_initialization() -> None:
    input_chart = Chart.linspace(8, low=-1.0, high=1.0)
    output_chart = Chart.linspace(5, low=-1.0, high=1.0)
    atoms = 4096
    kernels = (
        Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.7),
            )
        ),
        AmplitudeBandwidthSeparable(
            sigma_min=0.7,
            sigma_max=1.2,
        ),
    )

    for seed, kernel in enumerate(kernels):
        torch.manual_seed(seed)
        p = kernel.initialize(input_chart, output_chart, atoms, mode="uniform")
        expected = 0.1 / atoms**0.5
        assert abs(float(p[:, 0].mean())) < 0.05 * expected
        assert abs(float(p[:, 0].std()) - expected) < 0.05 * expected
    assert tuple(kernel.parameters()) == ()


def make_bandwidth_kernel() -> AmplitudeBandwidthSeparable:
    return AmplitudeBandwidthSeparable(
        sigma_min=0.1,
        sigma_max=0.8,
        tau=0.5,
        temperature=0.25,
        gate_eps=1e-12,
    )


def test_amplitude_bandwidth_interpolates_precision_at_the_threshold() -> None:
    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    kernel = make_bandwidth_kernel()
    p = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    gate = kernel.amplitude_gate(input_chart, output_chart, p)
    precision = kernel.bandwidth_precision(input_chart, output_chart, p)
    sigma = kernel.bandwidth_sigma(input_chart, output_chart, p)
    broad_precision = torch.tensor(1.0 / 0.8**2, dtype=p.dtype)
    narrow_precision = torch.tensor(1.0 / 0.1**2, dtype=p.dtype)

    torch.testing.assert_close(gate[1], torch.tensor(0.5, dtype=p.dtype))
    torch.testing.assert_close(
        precision[1],
        0.5 * (broad_precision + narrow_precision),
    )
    torch.testing.assert_close(sigma[0], torch.tensor(0.8, dtype=p.dtype))
    torch.testing.assert_close(sigma[2], torch.tensor(0.1, dtype=p.dtype))
    assert gate[0] < gate[1] < gate[2]


def test_weak_atom_uses_the_same_finite_broad_width_on_both_sides() -> None:
    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    kernel = make_bandwidth_kernel()
    p = torch.tensor([[1e-10, 0.1, -0.2]], dtype=torch.float64)

    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)
    precision = kernel.bandwidth_precision(input_chart, output_chart, p)
    broad = torch.tensor(1.0 / 0.8**2, dtype=p.dtype)

    torch.testing.assert_close(precision[0], broad)
    torch.testing.assert_close(
        phi_input,
        kernel.profile.evaluate_with_precision(input_chart, p[:, 1:2], broad),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        phi_output / p[0, 0],
        kernel.profile.evaluate_with_precision(output_chart, p[:, 2:], broad),
        atol=1e-12,
        rtol=0,
    )


def test_bandwidth_bounds_must_be_finite_and_ordered() -> None:
    for sigma_max in (float("inf"), float("nan"), 0.0):
        with pytest.raises(ValueError, match="sigma_max"):
            AmplitudeBandwidthSeparable(sigma_min=0.1, sigma_max=sigma_max)
    with pytest.raises(ValueError, match="sigma_max"):
        AmplitudeBandwidthSeparable(sigma_min=0.2, sigma_max=0.1)
    with pytest.raises(ValueError, match="law"):
        AmplitudeBandwidthSeparable(sigma_min=0.1, sigma_max=1.0, law="step")


def make_inverse_bandwidth_kernel() -> AmplitudeBandwidthSeparable:
    return AmplitudeBandwidthSeparable(
        sigma_min=0.01,
        sigma_max=100.0,
        tau=0.5,
        temperature=0.25,
        gate_eps=1e-12,
        law="inverse",
    )


def test_inverse_bandwidth_tracks_reciprocal_amplitude_in_the_interior() -> None:
    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    kernel = make_inverse_bandwidth_kernel()
    amplitude = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
    p = torch.stack((amplitude, torch.zeros_like(amplitude), torch.zeros_like(amplitude)), dim=1)

    sigma = kernel.bandwidth_sigma(input_chart, output_chart, p)
    expected = kernel.tau.to(dtype=p.dtype) / amplitude.abs()
    torch.testing.assert_close(sigma, expected, rtol=1e-5, atol=1e-8)
    assert torch.all(sigma[:-1] > sigma[1:])


def test_inverse_bandwidth_is_even_and_clamped() -> None:
    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    kernel = make_inverse_bandwidth_kernel()
    p = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1e-12, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0e4, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    sigma = kernel.bandwidth_sigma(input_chart, output_chart, p)
    torch.testing.assert_close(
        sigma[0],
        torch.tensor(100.0, dtype=p.dtype),
        rtol=1e-5,
        atol=1e-8,
    )
    torch.testing.assert_close(
        sigma[1],
        torch.tensor(100.0, dtype=p.dtype),
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(sigma[2], sigma[3], rtol=0, atol=0)
    torch.testing.assert_close(
        sigma[4],
        torch.tensor(0.01, dtype=p.dtype),
        rtol=1e-5,
        atol=1e-8,
    )


def test_inverse_bandwidth_precision_jacobian_matches_autograd() -> None:
    kernel = make_inverse_bandwidth_kernel()
    amplitude = torch.tensor([[0.0], [1e-10], [0.25], [0.5], [2.0]], dtype=torch.float64)
    _, dprecision = kernel._precision_and_jacobian(amplitude)
    autograd = torch.func.grad(
        lambda value: kernel._precision_and_jacobian(value)[0].sum()
    )(amplitude)
    torch.testing.assert_close(dprecision, autograd[:, 0])
    assert dprecision[0].abs() < 1e-18
    assert dprecision[2] > 0


@pytest.mark.parametrize(
    "kernel_factory",
    [make_bandwidth_kernel, make_inverse_bandwidth_kernel],
)
def test_amplitude_bandwidth_factorization_and_second_derivatives_are_finite(
    kernel_factory,
) -> None:
    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    kernel = kernel_factory()
    p = torch.tensor([[0.0, 0.1, -0.2]], dtype=torch.float64)

    represented = kernel.materialize_atoms(input_chart, output_chart, p)
    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)
    hessian = torch.func.hessian(
        lambda atom: kernel.materialize_atoms(
            input_chart,
            output_chart,
            atom.unsqueeze(0),
        ).sum()
    )(p[0])

    torch.testing.assert_close(
        represented,
        torch.einsum("oa,ia->aoi", phi_output, phi_input),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        p[:, 0].abs(),
    )
    assert torch.isfinite(hessian).all()
    assert torch.linalg.vector_norm(hessian[0, 1:]) > 0
    assert tuple(kernel.parameters()) == ()


def test_decoupled_bandwidth_keeps_forward_sigma_and_zeros_amplitude_width_grad() -> None:
    coupled = AmplitudeBandwidthSeparable(
        sigma_min=0.1,
        sigma_max=0.8,
        tau=0.5,
        temperature=0.25,
    )
    detached = AmplitudeBandwidthSeparable(
        sigma_min=0.1,
        sigma_max=0.8,
        tau=0.5,
        temperature=0.25,
        couple_bandwidth=False,
    )
    amplitude = torch.tensor([[0.5]], dtype=torch.float64)
    _, coupled_jac = coupled._precision_and_jacobian(amplitude)
    _, detached_jac = detached._precision_and_jacobian(amplitude)
    assert coupled_jac.abs() > 1e-6
    torch.testing.assert_close(
        detached_jac, torch.zeros_like(detached_jac), atol=0, rtol=0
    )

    input_chart = Chart.linspace(3, low=-1.0, high=1.0)
    output_chart = Chart.linspace(4, low=-1.0, high=1.0)
    p = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        coupled.bandwidth_sigma(input_chart, output_chart, p),
        detached.bandwidth_sigma(input_chart, output_chart, p),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        coupled.materialize_atoms(input_chart, output_chart, p),
        detached.materialize_atoms(input_chart, output_chart, p),
        atol=0,
        rtol=0,
    )
