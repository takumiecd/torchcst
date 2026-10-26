"""Positive coefficient formula versus the independent native FFT/convolution."""

import pytest
import torch

from torchcst._backends.torch.profiles.fourier import (
    chebyshev_square_coefficient,
    kernel_coefficients,
)
from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec, KaiserSquareSpec


@pytest.mark.parametrize("modes", [2, 4, 8, 16])
def test_positive_polynomial_coefficients_and_every_sigma_derivative(modes):
    profile = ChebyshevSquareSpec(modes=modes, tail_bound=0.1 if modes == 2 else 1e-3)
    sigma = torch.linspace(0.001, 16.0, 17, dtype=torch.float64, requires_grad=True)
    expected = kernel_coefficients(profile, sigma, 48.0)[:, modes:]
    for k in range(modes + 1):
        actual = chebyshev_square_coefficient(profile, sigma, 48.0, k)
        torch.testing.assert_close(actual, expected[:, k], rtol=5e-12, atol=5e-13)
        da = torch.autograd.grad(actual.sum(), sigma, retain_graph=True)[0]
        de = torch.autograd.grad(expected[:, k].sum(), sigma, retain_graph=True)[0]
        torch.testing.assert_close(da, de, rtol=5e-10, atol=5e-12)
        assert (actual >= 0).all()


def test_coefficient_formula_rejects_other_semantics():
    sigma = torch.ones(1, dtype=torch.float64)
    with pytest.raises(TypeError, match="ChebyshevSquareSpec"):
        chebyshev_square_coefficient(KaiserSquareSpec(modes=16), sigma, 48.0, 0)
    with pytest.raises(ValueError, match="harmonic"):
        chebyshev_square_coefficient(ChebyshevSquareSpec(modes=8), sigma, 48.0, 9)
