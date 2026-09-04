import pytest
import torch

from torchcst import Gaussian


def test_gaussian_evaluates_all_point_pairs() -> None:
    kernel = Gaussian(2.0)
    left = torch.tensor([[0.0], [2.0]])
    right = torch.tensor([[0.0], [1.0], [2.0]])

    actual = kernel(left, right)
    expected = torch.exp(-((left - right.T).square()) / 8.0)

    torch.testing.assert_close(actual, expected)


def test_trainable_bandwidth_is_positive_and_differentiable() -> None:
    kernel = Gaussian(0.5, trainable=True)
    value = kernel(torch.tensor([[0.0]]), torch.tensor([[1.0]])).sum()

    value.backward()

    assert kernel.sigma.item() > 0
    assert len(tuple(kernel.parameters())) == 1
    assert next(iter(kernel.parameters())).grad is not None


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf")])
def test_sigma_must_be_finite_and_positive(sigma: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Gaussian(sigma)
