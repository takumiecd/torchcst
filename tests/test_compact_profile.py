from __future__ import annotations

import pytest
import torch
from kernel_cases import (
    amp_width_state,
    biweight_state,
    gaussian_state,
    separable_state,
    triangle_state,
    triweight_state,
    wendland_state,
)
from torch.func import hessian, vmap

from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile


def _double_profile(factory, sigma: float):
    return factory(sigma).to(dtype=torch.float64)


def test_wendland_columns_are_l2_normalized_or_zero() -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 9).unsqueeze(-1).double())
    profile = _double_profile(wendland_state, 0.5)
    p = torch.tensor([[0.0], [8.0]], dtype=torch.float64)
    values = _profile.evaluate(profile, chart, p)
    norms = torch.linalg.vector_norm(values, dim=0)
    torch.testing.assert_close(norms[0], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(norms[1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.equal(values[:, 1], torch.zeros(chart.features, dtype=torch.float64))
    assert tuple(profile.parameters()) == ()


def test_wendland_matches_the_unnormalized_polynomial() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = _double_profile(wendland_state, 1.0)
    p = torch.tensor([[0.0]], dtype=torch.float64)
    raw = torch.tensor([1.0, 0.5**4 * 3.0, 0.0], dtype=torch.float64)
    expected = raw / torch.linalg.vector_norm(raw)
    torch.testing.assert_close(_profile.evaluate(profile, chart, p)[:, 0], expected)


def test_triweight_matches_the_unnormalized_polynomial() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = triweight_state(1.0, normalize_columns=False).double()
    p = torch.tensor([[0.0]], dtype=torch.float64)
    raw = torch.tensor([1.0, 0.75**3, 0.0], dtype=torch.float64)
    torch.testing.assert_close(_profile.evaluate(profile, chart, p)[:, 0], raw)


def test_triweight_defaults_to_legacy_l2_column_normalization() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = triweight_state(1.0).double()
    p = torch.tensor([[0.0]], dtype=torch.float64)
    raw = torch.tensor([1.0, 0.75**3, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        _profile.evaluate(profile, chart, p)[:, 0], raw / torch.linalg.vector_norm(raw)
    )


def test_triangle_matches_the_radial_hat() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = _double_profile(triangle_state, 1.0)
    p = torch.tensor([[0.0]], dtype=torch.float64)
    expected = torch.tensor(
        [1.0 - torch.finfo(torch.float64).eps ** 0.5, 0.5, 0.0], dtype=torch.float64
    )
    torch.testing.assert_close(_profile.evaluate(profile, chart, p)[:, 0], expected)


def test_forward_skips_offsets_but_analytic_tangent_uses_them(monkeypatch) -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 9).unsqueeze(-1))
    profile = triweight_state(0.5)
    centers = torch.tensor([[0.1]])
    original = _charts.center_offsets
    calls = 0

    def tracked(state, p):
        nonlocal calls
        calls += 1
        return original(state, p)

    monkeypatch.setattr(_charts, "center_offsets", tracked)
    _profile.evaluate(profile, chart, centers)
    assert calls == 0
    _profile.tangent(profile, chart, centers)
    assert calls == 1


def test_biweight_matches_the_unnormalized_polynomial() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = _double_profile(biweight_state, 1.0)
    p = torch.tensor([[0.0]], dtype=torch.float64)
    expected = torch.tensor([1.0, 0.75**2, 0.0], dtype=torch.float64)
    torch.testing.assert_close(_profile.evaluate(profile, chart, p)[:, 0], expected)


def test_biweight_can_opt_into_discrete_column_normalization() -> None:
    chart = _construction.points(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    )
    profile = biweight_state(1.0, normalize_columns=True).double()
    p = torch.tensor([[0.0]], dtype=torch.float64)
    raw = torch.tensor([1.0, 0.75**2, 0.0], dtype=torch.float64)
    expected = raw / torch.linalg.vector_norm(raw)
    torch.testing.assert_close(_profile.evaluate(profile, chart, p)[:, 0], expected)
    assert profile.binding.normalization.kind == "discrete_l2"


def test_compact_normalization_override_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="bool"):
        biweight_state(1.0, normalize_columns=1)


@pytest.mark.parametrize("factory", [triangle_state, biweight_state, triweight_state])
def test_raw_compact_profile_retains_one_site_center_and_width_derivatives(
    factory,
) -> None:
    chart = _construction.points(torch.tensor([[0.0], [2.0]], dtype=torch.float64))
    profile = factory(1.0, normalize_columns=False).double()
    center = torch.tensor([[0.25]], dtype=torch.float64)
    precision = torch.ones(1, dtype=torch.float64)
    values, centers, widths = _profile.tangent_with_precision(
        profile, chart, center, precision
    )
    assert torch.count_nonzero(values[:, 0]) == 1
    assert centers[0, 0, 0] != 0
    assert widths[0, 0] != 0
    torch.testing.assert_close(centers[1], torch.zeros_like(centers[1]))
    torch.testing.assert_close(widths[1], torch.zeros_like(widths[1]))


@pytest.mark.parametrize(
    "factory", [wendland_state, triangle_state, biweight_state, triweight_state]
)
def test_separated_compact_atoms_have_zero_inner_product(factory) -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    compact = _double_profile(factory, 0.35)
    gaussian = gaussian_state(0.35).to(dtype=torch.float64)
    p = torch.tensor([[-0.75], [0.75]], dtype=torch.float64)
    compact_values = _profile.evaluate(compact, chart, p)
    gaussian_values = _profile.evaluate(gaussian, chart, p)
    compact_gram = compact_values.T @ compact_values
    gaussian_gram = gaussian_values.T @ gaussian_values
    torch.testing.assert_close(
        compact_gram[0, 1], torch.zeros((), dtype=torch.float64), atol=0, rtol=0
    )
    assert gaussian_gram[0, 1].abs() > 1e-06


@pytest.mark.parametrize(
    "factory", [wendland_state, triangle_state, biweight_state, triweight_state]
)
def test_compact_tangent_is_finite_on_the_support_boundary(factory) -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
    values, centers, widths = _profile.tangent_with_precision(
        profile, chart, p, profile.sigma.reciprocal().square()
    )
    assert torch.isfinite(values).all()
    assert torch.isfinite(centers).all()
    assert torch.isfinite(widths).all()


def test_wendland_mnist_grid_tangent_is_finite_float32() -> None:
    chart = _construction.grid((28, 28), low=-1.0, high=1.0)
    profile = wendland_state(0.1)
    torch.manual_seed(17)
    p = _profile.initialize(profile, chart, 256, mode="uniform")
    values, centers, widths = _profile.tangent_with_precision(
        profile, chart, p, profile.sigma.reciprocal().square()
    )
    assert torch.isfinite(values).all()
    assert torch.isfinite(centers).all()
    assert torch.isfinite(widths).all()


@pytest.mark.parametrize("factory", [wendland_state, biweight_state, triweight_state])
def test_compact_autograd_hessian_is_finite_inside_boundary_and_outside(
    factory,
) -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 21).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.0], [0.5], [3.0]], dtype=torch.float64)

    def scalar(coords):
        return _profile.evaluate(profile, chart, coords.unsqueeze(0)).sum()

    blocks = vmap(hessian(scalar))(p)
    assert torch.isfinite(blocks).all()
    torch.testing.assert_close(
        blocks[2], torch.zeros(1, 1, dtype=torch.float64), atol=0, rtol=0
    )


def test_triweight_factor_hessian_is_finite_for_empty_atoms() -> None:
    kernel = amp_width_state(
        sigma_min=0.1, sigma_max=10.0, profile=triweight_state(0.1)
    )
    input_chart = _construction.grid((28, 28), low=-1.0, high=1.0)
    output_chart = _construction.linspace(64, low=-1.0, high=1.0)
    p = _kernel.initialize(kernel, input_chart, output_chart, 3, mode="uniform")
    p = p.clone()
    p[:, 0] = 1.0
    p[1, 1:3] = 8.0
    p[2, 1:3] = input_chart.coordinates[0]
    inputs = torch.randn(8, 784)
    output_gradient = torch.randn(8, 64)

    def scalar(atom):
        phi_in, phi_out = _kernel.factors(
            kernel, input_chart, output_chart, atom.unsqueeze(0)
        )
        return (inputs @ phi_in * (output_gradient @ phi_out)).sum()

    blocks = vmap(hessian(scalar))(p)
    assert torch.isfinite(blocks).all()


@pytest.mark.parametrize(
    "factory", [wendland_state, triangle_state, biweight_state, triweight_state]
)
def test_compact_tangent_matches_autograd(factory) -> None:
    chart = _construction.points(torch.linspace(-1.0, 1.0, 11).unsqueeze(-1).double())
    profile = _double_profile(factory, 0.5)
    p = torch.tensor([[0.0], [0.25]], dtype=torch.float64)
    precision = torch.tensor([4.0, 9.0], dtype=torch.float64)
    values, centers, widths = _profile.tangent_with_precision(
        profile, chart, p, precision
    )
    torch.testing.assert_close(
        values, _profile.evaluate_with_precision(profile, chart, p, precision)
    )
    jacobian_p = torch.autograd.functional.jacobian(
        lambda coords: _profile.evaluate_with_precision(
            profile, chart, coords, precision
        ),
        p,
    )
    jacobian_prec = torch.autograd.functional.jacobian(
        lambda prec: _profile.evaluate_with_precision(profile, chart, p, prec),
        precision,
    )
    for atom in range(p.shape[0]):
        torch.testing.assert_close(centers[:, atom, :], jacobian_p[:, atom, atom, :])
        torch.testing.assert_close(widths[:, atom], jacobian_prec[:, atom, atom])


@pytest.mark.parametrize(
    "factory", [wendland_state, triangle_state, biweight_state, triweight_state]
)
def test_compact_sigma_must_be_finite_and_positive(factory) -> None:
    for sigma in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            factory(sigma)


@pytest.mark.parametrize(
    "factory", [wendland_state, triangle_state, biweight_state, triweight_state]
)
def test_amplitude_bandwidth_accepts_compact_profiles(factory) -> None:
    kernel = amp_width_state(
        sigma_min=0.8, sigma_max=2.0, profile=factory(0.8)
    ).double()
    input_chart = _construction.points(
        torch.linspace(-1.0, 1.0, 9).unsqueeze(-1).double()
    )
    output_chart = _construction.points(
        torch.linspace(-1.0, 1.0, 7).unsqueeze(-1).double()
    )
    p = torch.zeros(2, 3, dtype=torch.float64)
    p[:, 0] = torch.tensor([0.4, -0.3], dtype=torch.float64)
    p[:, 1] = torch.tensor([0.0, 0.2], dtype=torch.float64)
    p[:, 2] = torch.tensor([0.0, -0.1], dtype=torch.float64)
    represented = _kernel.materialize_atoms(kernel, input_chart, output_chart, p)
    phi_input, phi_output = _kernel.factors(kernel, input_chart, output_chart, p)
    torch.testing.assert_close(
        represented, torch.einsum("oa,ia->aoi", phi_output, phi_input)
    )
    if factory is wendland_state:
        torch.testing.assert_close(
            torch.linalg.vector_norm(represented.flatten(1), dim=1), p[:, 0].abs()
        )
    hessian = torch.func.hessian(
        lambda atom: _kernel.materialize_atoms(
            kernel, input_chart, output_chart, atom.unsqueeze(0)
        ).sum()
    )(p[0])
    assert torch.isfinite(hessian).all()


def test_separable_accepts_compact_profiles() -> None:
    kernel = separable_state(
        input_profile=wendland_state(0.4), output_profile=triweight_state(0.3)
    )
    input_chart = _construction.linspace(4, low=-1.0, high=1.0)
    output_chart = _construction.linspace(3, low=-1.0, high=1.0)
    p = _kernel.initialize(kernel, input_chart, output_chart, 2, mode="balanced")
    phi_input, phi_output = _kernel.factors(kernel, input_chart, output_chart, p)
    assert phi_input.shape == (4, 2)
    assert phi_output.shape == (3, 2)
    assert _kernel.tangent_backend(kernel, input_chart, output_chart) is not None


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("center_value", [-(0.995**0.5), -0.99, -2.0])
def test_triweight_normalized_tangent_matches_clamped_norm_autograd(
    dtype, center_value
):
    chart = _construction.points(torch.tensor([[0.0]], dtype=dtype))
    profile = triweight_state(1.0).to(dtype=dtype)
    p = torch.tensor([[center_value]], dtype=dtype, requires_grad=True)
    precision = torch.ones(1, dtype=dtype, requires_grad=True)
    values, centers, widths = _profile.tangent_with_precision(
        profile, chart, p, precision
    )
    gc, gprec = torch.autograd.grad(values.sum(), (p, precision))
    torch.testing.assert_close(
        centers.sum(0), gc, atol=2e-05 if dtype == torch.float32 else 2e-12, rtol=2e-05
    )
    torch.testing.assert_close(
        widths.sum(0),
        gprec,
        atol=2e-05 if dtype == torch.float32 else 2e-12,
        rtol=2e-05,
    )
    assert torch.isfinite(centers).all() and torch.isfinite(widths).all()
