import torch

from torchcst import (
    Amplitude,
    AmplitudeBandwidthSeparable,
    Chart,
    Gaussian,
    Separable,
)


def charts() -> tuple[Chart, Chart]:
    return Chart.linspace(3), Chart.grid((2, 2))


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
    input_chart = Chart.linspace(8)
    output_chart = Chart.linspace(5)
    atoms = 4096
    kernels = (
        Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.7),
            )
        ),
        AmplitudeBandwidthSeparable(
            input_profile=Gaussian(0.8),
            output_profile=Gaussian(0.7),
        ),
    )

    for seed, kernel in enumerate(kernels):
        torch.manual_seed(seed)
        p = kernel.initialize(input_chart, output_chart, atoms, mode="uniform")
        expected = 0.1 / atoms**0.5
        assert abs(float(p[:, 0].mean())) < 0.05 * expected
        assert abs(float(p[:, 0].std()) - expected) < 0.05 * expected
    assert tuple(kernel.parameters()) == ()


def make_bandwidth_kernel(
    *, sigma_explore: float = 0.8
) -> AmplitudeBandwidthSeparable:
    return AmplitudeBandwidthSeparable(
        input_profile=Gaussian(0.4),
        output_profile=Gaussian(0.1),
        tau=0.5,
        temperature=0.25,
        sigma_explore=sigma_explore,
        gate_eps=1e-12,
    )


def test_amplitude_bandwidth_interpolates_precision_at_the_threshold() -> None:
    input_chart = Chart.linspace(3)
    output_chart = Chart.linspace(4)
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
    precision = kernel.output_precision(input_chart, output_chart, p)
    sigma = kernel.output_sigma(input_chart, output_chart, p)
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


def test_infinite_explorer_width_approaches_a_uniform_output_profile() -> None:
    input_chart = Chart.linspace(3)
    output_chart = Chart.linspace(4)
    kernel = make_bandwidth_kernel(sigma_explore=float("inf"))
    p = torch.tensor([[1e-10, 0.1, -0.2]], dtype=torch.float64)

    _, phi_output = kernel.factors(input_chart, output_chart, p)
    precision = kernel.output_precision(input_chart, output_chart, p)

    assert precision[0] < 1e-16
    torch.testing.assert_close(
        phi_output / p[0, 0],
        torch.full_like(phi_output, output_chart.features**-0.5),
        atol=1e-12,
        rtol=0,
    )


def test_amplitude_bandwidth_factorization_and_second_derivatives_are_finite() -> None:
    input_chart = Chart.linspace(3)
    output_chart = Chart.linspace(4)
    kernel = make_bandwidth_kernel()
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
