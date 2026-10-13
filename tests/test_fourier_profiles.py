"""Independent native-function and full-site operator qualification."""

import copy
import math

import numpy as np
import pytest
import torch

from torchcst import (
    BandwidthBounds,
    ChebyshevSquareSpec,
    CSTLinear,
    CSTOptimizer,
    DefaultRecipe,
    DPSSSquareSpec,
    ExecutionPlan,
    FixedSelector,
    KaiserSquareSpec,
    NormalizationSpec,
    chart_presets,
    make_algorithm_registry,
    presets,
)
from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.fourier import kernel_coefficients, values

FAMILIES = (ChebyshevSquareSpec, DPSSSquareSpec, KaiserSquareSpec)


def oracle_h(profile, sigma, period):
    """NumPy preparation, independent of Torch's batched implementation."""
    m, tile = profile.modes, period / 3
    if isinstance(profile, ChebyshevSquareSpec):
        radius0 = (
            period
            / math.pi
            * math.acos(1 / math.cosh(math.acosh(profile.tail_bound**-0.5) / m))
        )
    else:
        radius0 = period * profile.concentration_floor / (m + 1)
    radius = (radius0**4 + (1 - (radius0 / tile) ** 4) * sigma**4) ** 0.25
    if isinstance(profile, ChebyshevSquareSpec):
        beta = 1 / math.cos(math.pi * radius / period)
        theta = np.arange(m + 1) * 2 * math.pi / (m + 1)
        arg = beta * np.cos(theta / 2)
        h_values = np.where(
            np.abs(arg) <= 1,
            np.cos(m * np.arccos(np.clip(arg, -1, 1))),
            np.cosh(m * np.arccosh(np.maximum(np.abs(arg), 1))),
        )
        h_values /= math.cosh(m * math.acosh(beta))
        h = np.fft.fftshift(np.fft.fft(h_values)).real / (m + 1)
    elif isinstance(profile, KaiserSquareSpec):
        b = math.pi * (m + 1) * radius / period
        k = np.arange(-m // 2, m // 2 + 1)
        h = np.i0(b * np.sqrt(np.maximum(1 - (2 * k / m) ** 2, 0))) / np.i0(b)
    else:
        w = radius / period
        matrix = np.zeros((m + 1, m + 1))
        for j in range(m + 1):
            matrix[j, j] = ((m - 2 * j) / 2) ** 2 * math.cos(2 * math.pi * w)
        for j in range(1, m + 1):
            matrix[j - 1, j] = matrix[j, j - 1] = j * (m + 1 - j) / 2
        h = np.linalg.eigh(matrix)[1][:, -1]
    h = (h + h[::-1]) / 2
    return h / h.sum()


def model(cls, modes=32, floor=1e-6, regular=False, dtype=torch.float64):
    kernel = presets.polar_fourier_profile_product(
        profiles=(cls(modes=modes),) * 2,
        tile_widths=(10.0, 10.0),
        amplitude_max=1.0,
        bounds=BandwidthBounds(minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1),
        w_c=0.1,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    chart = (
        chart_presets.regular_grid(((700,), (500,)), spacing=0.1, geometry="flat_torus")
        if regular
        else chart_presets.periodic_grid(
            (700, 500), periods=(70.0, 50.0), origin=(-0.3, 0.2), output_dims=1
        )
    )
    p = torch.tensor(
        [
            [0.05, 1.0, -0.325, 50.175],
            [-0.08, 1.3, 9.65, 10.25],
            [0.0, 2.0, 69.675, -0.025],
            [0.2, 1.6, 30.12, 31.23],
        ],
        dtype=dtype,
    )
    return CSTLinear(
        chart=chart, kernel=kernel, atoms=p, dtype=dtype, backend="factored"
    )


def independent_weight(layer, p):
    """Enumerate sites and normalize each complete atom matrix once."""
    sigma = coordinates.bandwidth_sigma(layer.kernel, layer.chart, p).detach().numpy()
    amplitude = (
        layer.kernel.scalar("amplitude_max")
        * p[:, 0]
        / p[:, :2].square().sum(-1).sqrt()
    )
    chart = layer.chart
    sizes = (
        chart.coordinate_shape
        if hasattr(chart, "coordinate_shape")
        else chart.grid_shape
    )
    result = p.new_zeros(layer.out_features, layer.in_features)
    for a in range(len(p)):
        axes = []
        for dim, (size, period, origin, binding) in enumerate(
            zip(
                sizes,
                chart.geometry.periods.tolist(),
                chart.origin.tolist(),
                layer.kernel.spec.profiles,
            )
        ):
            center_number = float(p[a, dim + 2].detach()) - origin
            wraps = math.floor(center_number / period)
            owner = math.floor((center_number - wraps * period) / 10)
            tiles = round(period / 10)
            sites = size // tiles
            local_center = p[a, dim + 2] - origin - wraps * period - owner * 10
            axis = []
            h = p.new_tensor(oracle_h(binding.profile, float(sigma[a]), 30.0))
            frequencies = torch.arange(
                -binding.profile.modes // 2,
                binding.profile.modes // 2 + 1,
                dtype=p.dtype,
            )
            for j in range(size):
                tile, offset = divmod(j, sites)
                relative = (tile - owner) % tiles
                if relative == tiles - 1:
                    relative = -1
                if relative not in (-1, 0, 1):
                    axis.append(p.new_zeros(()))
                else:
                    spacing = (
                        float(chart.spacing[dim])
                        if hasattr(chart, "coordinate_shape")
                        else period / size
                    )
                    d = offset * spacing + 10 * relative - local_center
                    axis.append(
                        (h * torch.cos(frequencies * (2 * math.pi / 30) * d))
                        .sum()
                        .square()
                    )
            axes.append(torch.stack(axis))
        product = axes[0]
        for axis in axes[1:]:
            product = (product[:, None] * axis[None, :]).flatten()
        atom = product.reshape(layer.out_features, layer.in_features)
        result = result + amplitude[a] * atom / torch.linalg.vector_norm(
            atom
        ).clamp_min(layer.kernel.spec.normalization.floor)
    return result


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("modes", [16, 32, 64])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_native_definition_coefficients_and_complete_period_l2(cls, modes, sigma):
    profile = cls(modes=modes)
    h = oracle_h(profile, sigma, 30.0)
    c_expected = np.convolve(h, h)
    c = kernel_coefficients(profile, torch.tensor([sigma], dtype=torch.float64), 30.0)[
        0
    ]
    np.testing.assert_allclose(c.numpy(), c_expected, rtol=2e-11, atol=2e-13)
    k = np.arange(-modes // 2, modes // 2 + 1)
    norms = []
    for center in (0.0, 0.025, 0.05, 0.075, 0.1, 9.95):
        d = torch.arange(300, dtype=torch.float64) * 0.1 - center
        actual = values(
            profile, d[:, None], torch.tensor([sigma], dtype=torch.float64), 30.0
        )[:, 0]
        expected = (np.cos(d.numpy()[:, None] * (2 * math.pi / 30) * k) @ h) ** 2
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-9, atol=2e-12)
        assert torch.all(actual >= 0)
        norms.append(float(actual.square().sum()))
    np.testing.assert_allclose(
        norms, 300 * float(c.square().sum()), rtol=2e-11, atol=2e-12
    )


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("modes", [16, 32, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_direct_and_shared_full_values_vjp_against_independent_sites(cls, modes, floor):
    layer = model(cls, modes, floor)
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(
        2, 500, dtype=torch.float64, generator=generator, requires_grad=True
    )
    target = torch.randn(2, 700, dtype=torch.float64, generator=generator)
    p = layer.atoms.p
    oracle = x @ independent_weight(layer, p).T
    grads = torch.autograd.grad((oracle * target).sum(), (x, p))
    for shared in (False, True):
        layer.selector = (
            FixedSelector(
                ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe()),
                registry=make_algorithm_registry(),
            )
            if shared
            else None
        )
        y = layer(x)
        actual = torch.autograd.grad((y * target).sum(), (x, p))
        torch.testing.assert_close(y, oracle, rtol=2e-9, atol=2e-10)
        for value, truth in zip(actual, grads):
            torch.testing.assert_close(value, truth, rtol=2e-8, atol=2e-9)
        # Task width is detached; the Polar task gradient is angular only.
        torch.testing.assert_close(
            (actual[1][:, :2] * p[:, :2]).sum(-1),
            torch.zeros(4, dtype=p.dtype),
            rtol=0,
            atol=2e-10,
        )


@pytest.mark.parametrize("cls", FAMILIES)
def test_regular_grid_fp32_checkpoint_and_optimizer(cls):
    layer = model(cls, regular=True, dtype=torch.float32)
    x = torch.randn(2, 500)
    result = layer(x)
    assert torch.isfinite(result).all()
    replica = model(cls, regular=True, dtype=torch.float32)
    replica.load_state_dict(copy.deepcopy(layer.state_dict()))
    torch.testing.assert_close(replica(x), result)
    before = layer.atoms.p.detach().clone()
    optimizer = CSTOptimizer(
        torch.optim.AdamW(layer.parameters(), lr=1e-3), model=layer
    )
    result.square().sum().backward()
    optimizer.step()
    assert torch.isfinite(layer(x)).all() and not torch.equal(before, layer.atoms.p)
    incompatible = model(cls, modes=64, regular=True, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        incompatible.load_state_dict(replica.state_dict())


def test_alias_grid_has_exact_direct_l2_but_shared_shortcut_is_rejected():
    source = model(ChebyshevSquareSpec, modes=32)
    chart = chart_presets.periodic_grid((9, 9), periods=(30.0, 30.0), output_dims=1)
    layer = CSTLinear(
        chart=chart,
        kernel=source.kernel.spec,
        atoms=source.atoms.p.detach().clone(),
        dtype=torch.float64,
    )
    x = torch.randn(2, 9, dtype=torch.float64)
    assert torch.isfinite(layer(x)).all()
    layer.selector = FixedSelector(
        ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe()),
        registry=make_algorithm_registry(),
    )
    with pytest.raises(ValueError, match="more than 2M"):
        layer(x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"modes": True},
        {"modes": 17},
        {"modes": 0},
        {"modes": 1024},
        {"tail_bound": 0},
        {"tail_bound": float("nan")},
    ],
)
def test_invalid_chebyshev_declarations(kwargs):
    with pytest.raises(ValueError):
        ChebyshevSquareSpec(**kwargs)


def test_mixed_axis_profiles_and_live_period_refresh():
    source = model(ChebyshevSquareSpec)
    from dataclasses import replace

    from torchcst.kernels import ProfileBinding

    spec = replace(
        source.kernel.spec,
        profiles=(
            source.kernel.spec.profiles[0],
            ProfileBinding(profile=KaiserSquareSpec(modes=16)),
        ),
    )
    layer = CSTLinear(
        chart=source.chart,
        kernel=spec,
        atoms=source.atoms.p.detach().clone(),
        dtype=torch.float64,
        backend="factored",
    )
    # Valid reconfiguration changes sites-per-tile; both paths must refresh.
    with torch.no_grad():
        layer.chart.geometry.periods[0] = 140.0
        layer.chart.origin.add_(0.15)
    x = torch.randn(2, 500, dtype=torch.float64)
    direct = layer(x)
    layer.selector = FixedSelector(
        ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe()),
        registry=make_algorithm_registry(),
    )
    torch.testing.assert_close(layer(x), direct, rtol=2e-10, atol=2e-11)


def test_three_coordinate_product_and_full_operator_floor():
    source = model(ChebyshevSquareSpec, modes=16)
    from dataclasses import replace

    from torchcst import FourierTileSupportSpec

    kernel = replace(
        source.kernel.spec,
        profiles=source.kernel.spec.profiles + (source.kernel.spec.profiles[0],),
        tile_support=FourierTileSupportSpec(tile_widths=(10.0,) * 3),
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=1000.0
        ),
    )
    chart = chart_presets.periodic_grid(
        (30, 60, 90), periods=(30.0,) * 3, origin=(0.0, 0.0, 0.0), output_dims=2
    )
    layer = CSTLinear(
        chart=chart,
        kernel=kernel,
        atoms=torch.tensor([[0.1, 1.2, 0.05, 9.975, 29.95]], dtype=torch.float64),
        dtype=torch.float64,
        backend="factored",
    )
    x = torch.randn(2, 90, dtype=torch.float64, requires_grad=True)
    p = layer.atoms.p
    expected = x @ independent_weight(layer, p).T
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    truth = torch.autograd.grad(expected.square().sum(), (x, p))
    observed = torch.autograd.grad(actual.square().sum(), (x, p))
    for value, reference in zip(observed, truth):
        torch.testing.assert_close(value, reference, rtol=2e-9, atol=2e-10)


def test_exact_zero_outside_owner_triple_and_boundary_branch():
    layer = model(ChebyshevSquareSpec, modes=64)
    from torchcst._backends.torch.kernels.fourier_product import axis_geometry

    centers = torch.tensor(
        [[-0.3, 0.2], [9.7, 10.2], [69.7, 50.2]], dtype=torch.float64
    )
    _, mask = axis_geometry(layer.chart, centers, 0, 10.0, 700)
    assert torch.equal(mask.sum(0), torch.full((3,), 300))
    assert mask[0, 0] and mask[699, 0] and not mask[300, 0]
    # A boundary belongs to the tile on the right; a full global turn agrees.
    assert mask[100, 1] and not mask[699, 1]
    torch.testing.assert_close(mask[:, 0], mask[:, 2])


@pytest.mark.parametrize("size,period", [(600, 20.0), (700, 35.0), (701, 70.0)])
def test_nonintegral_or_partial_tile_layout_rejected(size, period):
    source = model(ChebyshevSquareSpec)
    chart = chart_presets.periodic_grid(
        (size, 500), periods=(period, 50.0), output_dims=1
    )
    with pytest.raises(ValueError):
        CSTLinear(
            chart=chart,
            kernel=source.kernel.spec,
            atoms=source.atoms.p.detach().clone(),
            dtype=torch.float64,
        )


def test_approximate_period_must_not_use_diagonal_norm_shortcut():
    layer = model(ChebyshevSquareSpec)
    with torch.no_grad():
        layer.chart.geometry.periods[0].add_(1e-7)
    x = torch.randn(2, 500, dtype=torch.float64)
    assert torch.isfinite(layer(x)).all()
    layer.selector = FixedSelector(
        ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe()),
        registry=make_algorithm_registry(),
    )
    with pytest.raises(ValueError, match="exact uniform tile period"):
        layer(x)
