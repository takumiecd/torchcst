import torch

from torchcst import Chart, CSTLinear, Gaussian, Separable
from torchcst._derivatives import DenseDerivativeOracle


def make_site() -> CSTLinear:
    torch.manual_seed(11)
    return CSTLinear(
        Chart.linspace(3),
        Chart.linspace(2),
        atoms=2,
        kernel=Separable(
            input_profile=Gaussian(0.8),
            output_profile=Gaussian(0.6),
        ),
        dtype=torch.float64,
        backend="factored",
    )


def test_current_point_is_atom_structured_and_opaque() -> None:
    site = make_site()
    derivatives = site.cst_derivatives()

    assert derivatives.current_point().shape == (
        site.atom_count,
        site.atoms.parameter_dim,
    )
    assert torch.equal(derivatives.current_point(), site.atoms.p)


def test_represented_operator_is_the_sum_of_complete_kernel_atoms() -> None:
    site = make_site()
    derivatives = site.cst_derivatives()
    parameter_point = derivatives.current_point()

    torch.testing.assert_close(
        derivatives.represented(parameter_point), site.dense_weight()
    )
    torch.testing.assert_close(
        derivatives.represented_atoms(parameter_point).sum(dim=0),
        site.dense_weight(),
    )


def test_matrix_free_derivatives_match_full_dense_oracle() -> None:
    site = make_site()
    actual = site.cst_derivatives()
    oracle = DenseDerivativeOracle(site.atoms, site._materialize_atoms)
    parameter_point = actual.current_point()
    left = torch.randn_like(parameter_point) * 0.1
    right = torch.randn_like(parameter_point) * 0.1
    at = torch.randn_like(parameter_point) * 0.1
    cotangent = torch.randn(
        site.out_features, site.in_features, dtype=parameter_point.dtype
    )
    jacobian = oracle.jacobian(parameter_point=parameter_point)
    hessian = oracle.hessian(parameter_point=parameter_point)

    expected_jvp = torch.einsum("oikq,kq->oi", jacobian, right)
    expected_second = torch.einsum("oikqlr,kq,lr->oi", hessian, left, right)
    full_contracted = oracle.full_contracted_hessian(
        cotangent, parameter_point=parameter_point
    )
    expected_blocks = torch.stack(
        [full_contracted[index, :, index, :] for index in range(site.atom_count)]
    )

    torch.testing.assert_close(
        actual.jvp(right, parameter_point=parameter_point), expected_jvp
    )
    torch.testing.assert_close(
        actual.second(left, right, parameter_point=parameter_point), expected_second
    )
    torch.testing.assert_close(
        actual.contracted_hessian(cotangent, parameter_point=parameter_point),
        expected_blocks,
    )
    torch.testing.assert_close(
        actual.pushforward(right, at=at, parameter_point=parameter_point),
        expected_jvp + torch.einsum("oikqlr,kq,lr->oi", hessian, at, right),
    )


def test_distinct_atom_hessian_blocks_are_exactly_zero() -> None:
    site = make_site()
    oracle = DenseDerivativeOracle(site.atoms, site._materialize_atoms)
    cotangent = torch.randn(
        site.out_features, site.in_features, dtype=site.atoms.p.dtype
    )
    full = oracle.full_contracted_hessian(cotangent)

    cross = full[0, :, 1, :]

    torch.testing.assert_close(cross, torch.zeros_like(cross))


def test_pushforward_and_pullback_are_adjoint() -> None:
    site = make_site()
    derivatives = site.cst_derivatives()
    parameter_point = derivatives.current_point()
    at = torch.randn_like(parameter_point) * 0.1
    vector = torch.randn_like(parameter_point)
    cotangent = torch.randn(
        site.out_features, site.in_features, dtype=parameter_point.dtype
    )

    visible_inner = (
        derivatives.pushforward(vector, at=at, parameter_point=parameter_point)
        * cotangent
    ).sum()
    parameter_inner = (
        vector * derivatives.pullback(cotangent, at=at, parameter_point=parameter_point)
    ).sum()

    torch.testing.assert_close(visible_inner, parameter_inner)


def test_displacement_is_the_second_order_taylor_map() -> None:
    site = make_site()
    derivatives = site.cst_derivatives()
    parameter_point = derivatives.current_point()
    direction = torch.randn_like(parameter_point)
    scale = 1e-3

    exact_change = derivatives.represented(
        parameter_point + scale * direction
    ) - derivatives.represented(parameter_point)
    quadratic_change = derivatives.displacement(
        scale * direction, parameter_point=parameter_point
    )

    assert torch.linalg.vector_norm(exact_change - quadratic_change) < 1e-8


def test_zero_displacement_pullback_matches_parameter_autograd() -> None:
    site = make_site()
    derivatives = site.cst_derivatives()
    cotangent = torch.randn(
        site.out_features, site.in_features, dtype=site.atoms.p.dtype
    )

    (site.dense_weight() * cotangent).sum().backward()
    expected = site.atoms.p.grad

    torch.testing.assert_close(derivatives.pullback(cotangent), expected)


def test_derivatives_reject_trainable_charts_for_now() -> None:
    site = CSTLinear(
        Chart.linspace(3, trainable=True),
        Chart.linspace(2),
        atoms=2,
        kernel=Separable(
            input_profile=Gaussian(0.5),
            output_profile=Gaussian(0.5),
        ),
    )

    try:
        site.cst_derivatives()
    except ValueError as error:
        assert "frozen charts only" in str(error)
    else:
        raise AssertionError("trainable charts must be rejected")
