import pytest
import torch
from torch.func import hessian, vmap

from torchcst import (
    AmplitudeBandwidthSeparable,
    Chart,
    Gaussian,
    Separable,
    Triweight,
    WendlandC2,
)


def _double_profile(factory, sigma: float):
    return factory(sigma).to(dtype=torch.float64)


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_compact_columns_are_l2_normalized_or_zero(factory) -> None:
    chart = Chart.points(torch.linspace(-1.0, 1.0, 9).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.0], [8.0]], dtype=torch.float64)

    values = profile.evaluate(chart, p)
    norms = torch.linalg.vector_norm(values, dim=0)

    torch.testing.assert_close(norms[0], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(norms[1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.equal(values[:, 1], torch.zeros(chart.features, dtype=torch.float64))
    assert tuple(profile.parameters()) == ()


def test_wendland_matches_the_unnormalized_polynomial() -> None:
    chart = Chart.points(torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64))
    profile = _double_profile(WendlandC2, 1.0)
    p = torch.tensor([[0.0]], dtype=torch.float64)

    raw = torch.tensor([1.0, (0.5**4) * 3.0, 0.0], dtype=torch.float64)
    expected = raw / torch.linalg.vector_norm(raw)
    torch.testing.assert_close(profile.evaluate(chart, p)[:, 0], expected)


def test_triweight_matches_the_unnormalized_polynomial() -> None:
    chart = Chart.points(torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64))
    profile = _double_profile(Triweight, 1.0)
    p = torch.tensor([[0.0]], dtype=torch.float64)

    raw = torch.tensor([1.0, 0.75**3, 0.0], dtype=torch.float64)
    expected = raw / torch.linalg.vector_norm(raw)
    torch.testing.assert_close(profile.evaluate(chart, p)[:, 0], expected)


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_separated_compact_atoms_have_zero_inner_product(factory) -> None:
    chart = Chart.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    compact = _double_profile(factory, 0.35)
    gaussian = Gaussian(0.35).to(dtype=torch.float64)
    p = torch.tensor([[-0.75], [0.75]], dtype=torch.float64)

    compact_values = compact.evaluate(chart, p)
    gaussian_values = gaussian.evaluate(chart, p)
    compact_gram = compact_values.T @ compact_values
    gaussian_gram = gaussian_values.T @ gaussian_values

    torch.testing.assert_close(
        compact_gram[0, 1],
        torch.zeros((), dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert gaussian_gram[0, 1].abs() > 1e-6


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_compact_tangent_is_finite_on_the_support_boundary(factory) -> None:
    chart = Chart.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)

    values, centers, widths = profile.tangent_with_precision(
        chart, p, profile.sigma.reciprocal().square()
    )

    assert torch.isfinite(values).all()
    assert torch.isfinite(centers).all()
    assert torch.isfinite(widths).all()


def test_wendland_mnist_grid_tangent_is_finite_float32() -> None:
    chart = Chart.grid((28, 28))
    profile = WendlandC2(0.10)
    torch.manual_seed(17)
    p = profile.initialize(chart, 256, mode="uniform")

    values, centers, widths = profile.tangent_with_precision(
        chart, p, profile.sigma.reciprocal().square()
    )

    assert torch.isfinite(values).all()
    assert torch.isfinite(centers).all()
    assert torch.isfinite(widths).all()


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_compact_autograd_hessian_is_finite_inside_boundary_and_outside(
    factory,
) -> None:
    chart = Chart.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.0], [0.5], [3.0]], dtype=torch.float64)

    def scalar(coords):
        return profile.evaluate(chart, coords.unsqueeze(0)).sum()

    blocks = vmap(hessian(scalar))(p)
    assert torch.isfinite(blocks).all()
    torch.testing.assert_close(
        blocks[2], torch.zeros(1, 1, dtype=torch.float64), atol=0, rtol=0
    )


def test_triweight_factor_hessian_is_finite_for_empty_atoms() -> None:
    kernel = AmplitudeBandwidthSeparable(
        sigma_min=0.10,
        sigma_max=10.0,
        profile=Triweight(0.10),
    )
    input_chart = Chart.grid((28, 28))
    output_chart = Chart.linspace(64)
    p = kernel.initialize(input_chart, output_chart, 3, mode="uniform")
    p = p.clone()
    p[:, 0] = 1.0
    p[1, 1:3] = 8.0
    p[2, 1:3] = input_chart.coordinates[0]
    inputs = torch.randn(8, 784)
    output_gradient = torch.randn(8, 64)

    def scalar(atom):
        phi_in, phi_out = kernel.factors(
            input_chart, output_chart, atom.unsqueeze(0)
        )
        return ((inputs @ phi_in) * (output_gradient @ phi_out)).sum()

    blocks = vmap(hessian(scalar))(p)
    assert torch.isfinite(blocks).all()


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_compact_tangent_matches_autograd(factory) -> None:
    chart = Chart.points(torch.linspace(-1.0, 1.0, 11).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.0], [0.25]], dtype=torch.float64)
    precision = torch.tensor([4.0, 9.0], dtype=torch.float64)

    values, centers, widths = profile.tangent_with_precision(chart, p, precision)
    torch.testing.assert_close(
        values, profile.evaluate_with_precision(chart, p, precision)
    )

    jacobian_p = torch.autograd.functional.jacobian(
        lambda coords: profile.evaluate_with_precision(chart, coords, precision),
        p,
    )
    jacobian_prec = torch.autograd.functional.jacobian(
        lambda prec: profile.evaluate_with_precision(chart, p, prec),
        precision,
    )
    for atom in range(p.shape[0]):
        torch.testing.assert_close(centers[:, atom, :], jacobian_p[:, atom, atom, :])
        torch.testing.assert_close(widths[:, atom], jacobian_prec[:, atom, atom])


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_compact_sigma_must_be_finite_and_positive(factory) -> None:
    for sigma in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            factory(sigma)


@pytest.mark.parametrize("factory", [WendlandC2, Triweight])
def test_amplitude_bandwidth_accepts_compact_profiles(factory) -> None:
    kernel = AmplitudeBandwidthSeparable(
        sigma_min=0.80,
        sigma_max=2.0,
        profile=factory(0.80),
    ).double()
    input_chart = Chart.points(torch.linspace(-1.0, 1.0, 9).unsqueeze(-1).double())
    output_chart = Chart.points(torch.linspace(-1.0, 1.0, 7).unsqueeze(-1).double())
    p = torch.zeros(2, 3, dtype=torch.float64)
    p[:, 0] = torch.tensor([0.4, -0.3], dtype=torch.float64)
    p[:, 1] = torch.tensor([0.0, 0.2], dtype=torch.float64)
    p[:, 2] = torch.tensor([0.0, -0.1], dtype=torch.float64)
    represented = kernel.materialize_atoms(input_chart, output_chart, p)
    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)

    torch.testing.assert_close(
        represented,
        torch.einsum("oa,ia->aoi", phi_output, phi_input),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(represented.flatten(1), dim=1),
        p[:, 0].abs(),
    )
    hessian = torch.func.hessian(
        lambda atom: kernel.materialize_atoms(
            input_chart,
            output_chart,
            atom.unsqueeze(0),
        ).sum()
    )(p[0])
    assert torch.isfinite(hessian).all()


def test_bandwidth_profile_sigma_must_match_sigma_min() -> None:
    with pytest.raises(ValueError, match="profile.sigma must match sigma_min"):
        AmplitudeBandwidthSeparable(
            sigma_min=0.10,
            sigma_max=1.0,
            profile=WendlandC2(0.20),
        )


def test_separable_accepts_compact_profiles() -> None:
    kernel = Separable(
        input_profile=WendlandC2(0.4),
        output_profile=Triweight(0.3),
    )
    input_chart = Chart.linspace(4)
    output_chart = Chart.linspace(3)
    p = kernel.initialize(input_chart, output_chart, 2, mode="balanced")
    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)
    assert phi_input.shape == (4, 2)
    assert phi_output.shape == (3, 2)
    assert kernel.tangent_backend(input_chart, output_chart) is not None
