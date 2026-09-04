import torch

from torchcst import Chart, CSTLinear, Gaussian
from torchcst._derivatives import DenseDerivativeOracle, LinearDerivatives


def make_site(*, trainable_kernel: bool = False) -> CSTLinear:
    torch.manual_seed(11)
    return CSTLinear(
        Chart.linspace(3),
        Chart.linspace(2),
        atoms=2,
        input_kernel=Gaussian(0.8, trainable=trainable_kernel),
        output_kernel=Gaussian(0.6),
        dtype=torch.float64,
        backend="factored",
    )


def test_parameter_layout_includes_each_owned_parameter_once() -> None:
    site = make_site(trainable_kernel=True)
    derivatives = LinearDerivatives(site)

    assert derivatives.numel == 2 + 2 + 2 + 1
    assert len({id(parameter) for parameter in derivatives.parameters}) == 4
    assert [spec.name for spec in derivatives.layout.specs] == [
        "source",
        "target",
        "amplitude",
        "input_kernel._log_sigma",
    ]


def test_shared_trainable_kernel_is_one_coordinate_affecting_both_sides() -> None:
    kernel = Gaussian(0.5, trainable=True)
    site = CSTLinear(
        Chart.linspace(3),
        Chart.grid((2, 2)),
        atoms=2,
        input_kernel=kernel,
        dtype=torch.float64,
    )
    derivatives = LinearDerivatives(site)
    point = derivatives.point()
    changed = point.clone()
    changed[-1] += 0.2

    assert [spec.name for spec in derivatives.layout.specs].count(
        "input_kernel._log_sigma"
    ) == 1
    assert not torch.equal(
        derivatives.represented(changed), derivatives.represented(point)
    )


def test_matrix_free_derivatives_match_dense_autograd_oracle() -> None:
    site = make_site(trainable_kernel=True)
    actual = LinearDerivatives(site)
    oracle = DenseDerivativeOracle(site)
    point = actual.point()
    left = torch.randn_like(point) * 0.1
    right = torch.randn_like(point) * 0.1
    at = torch.randn_like(point) * 0.1
    cotangent = torch.randn(site.out_features, site.in_features, dtype=point.dtype)

    torch.testing.assert_close(
        actual.jvp(right, point=point), oracle.jvp(right, point=point)
    )
    torch.testing.assert_close(
        actual.second(left, right, point=point),
        oracle.second(left, right, point=point),
    )
    torch.testing.assert_close(
        actual.displacement(right, point=point),
        oracle.displacement(right, point=point),
    )
    torch.testing.assert_close(
        actual.pushforward(right, at=at, point=point),
        oracle.pushforward(right, at=at, point=point),
    )
    torch.testing.assert_close(
        actual.pullback(cotangent, at=at, point=point),
        oracle.pullback(cotangent, at=at, point=point),
    )


def test_pushforward_and_pullback_are_adjoint() -> None:
    site = make_site()
    derivatives = LinearDerivatives(site)
    point = derivatives.point()
    at = torch.randn_like(point) * 0.1
    vector = torch.randn_like(point)
    cotangent = torch.randn(site.out_features, site.in_features, dtype=point.dtype)

    visible_inner = (derivatives.pushforward(vector, at=at) * cotangent).sum()
    parameter_inner = (vector * derivatives.pullback(cotangent, at=at)).sum()

    torch.testing.assert_close(visible_inner, parameter_inner)


def test_displacement_is_the_second_order_taylor_map() -> None:
    site = make_site()
    derivatives = LinearDerivatives(site)
    point = derivatives.point()
    direction = torch.randn_like(point)
    scale = 1e-3

    exact_change = derivatives.represented(point + scale * direction) - derivatives.represented(
        point
    )
    quadratic_change = derivatives.displacement(scale * direction, point=point)

    assert torch.linalg.vector_norm(exact_change - quadratic_change) < 1e-8


def test_derivative_engine_rejects_trainable_charts_for_now() -> None:
    site = CSTLinear(
        Chart.linspace(3, trainable=True),
        Chart.linspace(2),
        atoms=2,
        input_kernel=Gaussian(0.5),
    )

    try:
        LinearDerivatives(site)
    except ValueError as error:
        assert "frozen charts only" in str(error)
    else:
        raise AssertionError("trainable charts must be rejected")
