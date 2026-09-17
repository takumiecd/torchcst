import pytest
import torch

from torchcst import Chart, Gaussian


def test_gaussian_evaluates_chart_against_opaque_profile_coordinates() -> None:
    chart = Chart.points(torch.tensor([[0.0], [2.0]]))
    p = torch.tensor([[0.0], [1.0], [2.0]])
    profile = Gaussian(2.0)

    actual = profile.evaluate(chart, p)
    expected = torch.exp(-((chart.coordinates - p.T).square()) / 8.0)
    expected = expected / torch.linalg.vector_norm(expected, dim=0, keepdim=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(actual, dim=0),
        torch.ones(p.shape[0]),
    )
    assert tuple(profile.parameters()) == ()


def test_gaussian_initialization_is_profile_owned() -> None:
    chart = Chart.linspace(3, low=-1.0, high=1.0)
    profile = Gaussian(0.5)

    balanced = profile.initialize(chart, 2, mode="balanced")
    uniform = profile.initialize(chart, 20, mode="uniform")

    assert torch.equal(balanced, torch.tensor([[-1.0], [1.0]]))
    assert bool((uniform >= -1.0).all())
    assert bool((uniform <= 1.0).all())


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf")])
def test_sigma_must_be_finite_and_positive(sigma: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Gaussian(sigma)
