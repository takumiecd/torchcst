import pytest
import torch

from torchcst import Chart, CSTLinear, Gaussian, Separable
from torchcst._derivatives import CSTSite


def make_site(*, backend: str = "factored") -> CSTLinear:
    return CSTLinear(
        Chart.linspace(5),
        Chart.grid((2, 2)),
        atoms=3,
        kernel=Separable(
            input_profile=Gaussian(0.4),
            output_profile=Gaussian(0.7),
        ),
        backend=backend,
        dtype=torch.float64,
    )


@pytest.mark.parametrize("backend", ["factored", "materialized"])
def test_capture_matches_the_dense_linear_cotangent(backend: str) -> None:
    torch.manual_seed(3)
    site = make_site(backend=backend)
    inputs = torch.randn(2, 3, site.in_features, dtype=torch.float64)
    output_cotangent = torch.randn(2, 3, site.out_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    (site(inputs) * output_cotangent).sum().backward()

    expected = output_cotangent.reshape(-1, site.out_features).T @ inputs.reshape(
        -1, site.in_features
    )
    torch.testing.assert_close(site.represented_gradient(), expected)


def test_capture_accumulates_repeated_calls_and_backward_passes() -> None:
    torch.manual_seed(5)
    site = make_site()
    x1 = torch.randn(4, site.in_features, dtype=torch.float64)
    x2 = torch.randn(2, site.in_features, dtype=torch.float64)
    g1 = torch.randn(4, site.out_features, dtype=torch.float64)
    g2 = torch.randn(2, site.out_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    (site(x1) * g1).sum().backward()
    (site(x2) * g2).sum().backward()

    torch.testing.assert_close(site.represented_gradient(), g1.T @ x1 + g2.T @ x2)


def test_derivative_pullback_of_capture_matches_parameter_gradients() -> None:
    torch.manual_seed(9)
    site = make_site()
    inputs = torch.randn(6, site.in_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    site(inputs).square().mean().backward()

    expected = torch.cat(
        (site.atoms.weight.grad.unsqueeze(-1), site.atoms.p.grad), dim=-1
    )
    actual = site.cst_derivatives().pullback(site.represented_gradient())
    torch.testing.assert_close(actual, expected)


def test_clear_and_disable_define_an_explicit_lifecycle() -> None:
    site = make_site()
    inputs = torch.randn(2, site.in_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    outputs = site(inputs)
    site.disable_represented_gradient_capture()
    outputs.sum().backward()

    with pytest.raises(RuntimeError, match="no represented gradient"):
        site.represented_gradient()

    site.enable_represented_gradient_capture()
    site(inputs).sum().backward()
    site.clear_represented_gradient()

    with pytest.raises(RuntimeError, match="no represented gradient"):
        site.represented_gradient()


def test_reenabled_capture_ignores_hooks_from_an_old_scope() -> None:
    site = make_site()
    old_inputs = torch.randn(2, site.in_features, dtype=torch.float64)
    new_inputs = torch.randn(3, site.in_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    old_outputs = site(old_inputs)
    site.disable_represented_gradient_capture()
    site.enable_represented_gradient_capture()
    new_outputs = site(new_inputs)
    (old_outputs.sum() + new_outputs.sum()).backward()

    expected = torch.ones(site.out_features, 3, dtype=torch.float64) @ new_inputs
    torch.testing.assert_close(site.represented_gradient(), expected)


def test_capture_is_transient_and_returns_an_isolated_snapshot() -> None:
    site = make_site()
    state_keys = tuple(site.state_dict())
    inputs = torch.randn(2, site.in_features, dtype=torch.float64)

    site.enable_represented_gradient_capture()
    site(inputs).sum().backward()
    snapshot = site.represented_gradient()
    snapshot.zero_()

    assert torch.count_nonzero(site.represented_gradient()) > 0
    assert tuple(site.state_dict()) == state_keys
    assert all("represented_gradient" not in key for key in state_keys)


def test_cst_linear_satisfies_the_site_protocol() -> None:
    site = make_site()

    assert isinstance(site, CSTSite)
    assert site.cst_parameters() == (site.atoms.weight, site.atoms.p)
