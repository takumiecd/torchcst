"""Check radius-cell Conv2d against independent scalar quadrature."""

import math

import pytest
import torch
import torch.nn.functional as F
from dispatch_cases import fixed_plan
from test_radius_cell_mass import oracle_mass

from torchcst import ChebyshevSpec, CSTConv2d, chart_presets, kernels
from torchcst._backends.torch.algorithms.conv2d import (
    TORCH_COMPOSED_CONV2D,
)
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
    ],
)
def test_composed_rational_cell_mass_matches_quadrature(kernel_size, device):
    patch = 2 * kernel_size**2
    chart = chart_presets.regular_grid
    output, input_ = (
        chart(4, geometry="flat_torus"),
        chart(patch, geometry="flat_torus"),
    )
    profile = ChebyshevSpec(modes=8)
    low, birth, high = 0.15, 0.8, 0.95 if kernel_size == 1 else 1.7
    kernel = kernels.product(
        kernels.cell_mass(profile),
        kernels.cell_mass(profile, base_tile_sites=kernel_size**2),
        coordinates=kernels.rational_radius(
            radius_min=low, radius_birth=birth, radius_max=high
        ),
    )
    conv = CSTConv2d(
        charts=(output, input_),
        in_channels=2,
        out_channels=4,
        kernel_size=kernel_size,
        stride=2,
        padding=kernel_size // 2,
        atoms=3,
        kernel=kernel,
        selector=fixed_plan(TORCH_COMPOSED_CONV2D),
        dtype=torch.float64,
    ).to(device)
    rows = torch.tensor(
        [
            [0.19, 0.31, 3.72, patch - 0.28],
            [-0.32, 0.49, 0.22, 0.39],
            [0.1, 0.21, 1.13, 1.19],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        conv.atoms.p.copy_(rows)

    axes = radius_cell_mass_axes((output, input_), kernel)

    def weight(parameters):
        weights = torch.zeros(4, patch, dtype=torch.float64)
        for amplitude, q, out_center, in_center in parameters.tolist():
            lower = low + (birth - low) / (1 + 3 * (amplitude / 0.3) ** 2)
            radius = math.exp((1 - q) * math.log(lower) + q * math.log(high))
            fo = torch.from_numpy(oracle_mass(axes[0], radius, out_center))
            fi = torch.from_numpy(oracle_mass(axes[1], radius, in_center))
            weights.add_(torch.outer(fo, fi), alpha=amplitude)
        return weights.view(4, 2, kernel_size, kernel_size)

    generator = torch.Generator().manual_seed(31)
    x_ref = torch.randn(2, 2, 7, 9, dtype=torch.float64, generator=generator)
    x_ref.requires_grad_()

    def reference(parameters):
        return F.conv2d(x_ref, weight(parameters), stride=2, padding=kernel_size // 2)

    expected = reference(rows)
    dy = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    (dx_ref,) = torch.autograd.grad(expected, x_ref, dy)
    dp_ref = torch.empty_like(rows)
    epsilon = 1e-5
    with torch.no_grad():
        for i in range(rows.numel()):
            delta = torch.zeros_like(rows)
            delta.view(-1)[i] = epsilon
            difference = reference(rows + delta) - reference(rows - delta)
            dp_ref.view(-1)[i] = (difference * dy).sum() / (2 * epsilon)

    x = x_ref.detach().to(device).requires_grad_()
    y = conv(x)
    dx, dp = torch.autograd.grad(y, (x, conv.atoms.p), dy.to(device))
    torch.testing.assert_close(y.cpu(), expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(dx.cpu(), dx_ref, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(dp.cpu(), dp_ref, atol=1e-8, rtol=1e-7)
