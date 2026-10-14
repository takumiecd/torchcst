"""Independent native tables, actual-site L2, and two-tile branch checks."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    FourierTwoTileSupportSpec,
    KaiserSquareSpec,
    KaiserTemplateSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)
from torchcst._backends.torch.kernels.fourier_two_tile_product import axis_geometry
from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.two_tile_template import (
    kernel_coefficients,
    static_tables,
    values,
)
from torchcst.kernels.fourier_layout import fourier_tile_layout
from torchcst.kernels.fourier_two_tile_layout import two_tile_layout


def native_tables(spec, period):
    """NumPy I0 and linear convolution, independent of the Torch preparation."""
    m = spec.modes
    k = np.arange(-m // 2, m // 2 + 1)
    r0, radius = spec.concentration_floor * period / (m + 1), period / 4
    result = []
    for sigma in spec.width_knots:
        r = (r0**4 + (1 - (r0 / radius) ** 4) * sigma**4) ** 0.25
        beta = math.pi * (m + 1) * r / period
        h = np.i0(beta * np.sqrt(np.maximum(1 - (2 * k / m) ** 2, 0))) / np.i0(beta)
        result.append(h / h.sum())
    h = np.array(result)
    return h, np.array([np.convolve(row, row) for row in h])


@pytest.mark.parametrize("modes", [16, 24, 32])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3, 0.7, 1.0, 2.0, 6.0, 8.0, 10.0])
def test_positive_native_blend_coefficients_and_shifted_period_l2(modes, sigma):
    spec = KaiserTemplateSpec(modes=modes)
    h_expected, c_expected = native_tables(spec, 40)
    h, c, gram = static_tables(spec, 40)
    np.testing.assert_allclose(h.numpy(), h_expected, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(c.numpy(), c_expected, rtol=2e-13, atol=2e-15)
    expected_gram = np.stack(
        (
            np.sum(c_expected[:-1] ** 2, axis=-1),
            np.sum(c_expected[:-1] * c_expected[1:], axis=-1),
            np.sum(c_expected[1:] ** 2, axis=-1),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(gram.numpy(), expected_gram, rtol=2e-13, atol=2e-15)
    j = min(
        np.searchsorted(spec.width_knots, sigma, side="right") - 1, len(h_expected) - 2
    )
    t = math.log(sigma / spec.width_knots[j]) / math.log(
        spec.width_knots[j + 1] / spec.width_knots[j]
    )
    expected_coeff = (1 - t) * c_expected[j] + t * c_expected[j + 1]
    sigma_tensor = torch.tensor([sigma], dtype=torch.float64)
    coeff = kernel_coefficients(spec, sigma_tensor, 40)[0]
    np.testing.assert_allclose(coeff.numpy(), expected_coeff, rtol=2e-13, atol=2e-15)
    frequencies = np.arange(-modes // 2, modes // 2 + 1)
    for center in (0.0, 0.025, 0.05, 0.075, 10.0, 19.975, 20.0):
        d = torch.arange(400, dtype=torch.float64) * 0.1 - center
        observed = values(spec, d[:, None], sigma_tensor, 40)[:, 0]
        basis = np.cos(d.numpy()[:, None] * 2 * math.pi / 40 * frequencies)
        expected = (1 - t) * (basis @ h_expected[j]) ** 2 + t * (
            basis @ h_expected[j + 1]
        ) ** 2
        np.testing.assert_allclose(observed.numpy(), expected, rtol=3e-11, atol=3e-14)
        assert bool((observed >= 0).all())
        torch.testing.assert_close(
            observed.square().sum(), 400 * coeff.square().sum(), rtol=2e-13, atol=2e-12
        )


def model(modes=16, *, floor=1e-6, regular=False, dtype=torch.float64):
    kernel = presets.polar_two_tile_template_product(
        profiles=(KaiserTemplateSpec(modes=modes),) * 2,
        tile_widths=(20.0,) * 2,
        amplitude_max=1.0,
        w_c=0.1,
        bounds=BandwidthBounds(minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1),
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    chart = (
        chart_presets.regular_grid(((400,), (400,)), spacing=0.2, geometry="flat_torus")
        if regular
        else chart_presets.periodic_grid(
            (400, 400), periods=(80.0, 80.0), origin=(-0.3, 0.2), output_dims=1
        )
    )
    return CSTLinear(
        chart=chart,
        kernel=kernel,
        atoms=torch.tensor(
            [
                [0.1, 1.0, -0.275, 80.175],
                [-0.2, 1.3, 9.675, 10.225],
                [0.3, 1.6, 19.725, 20.175],
                [-0.1, 1.8, 79.675, 60.225],
            ],
            dtype=dtype,
        ),
        dtype=dtype,
        backend="factored",
    )


def independent_weight(layer, p):
    """Native per-site masks and one floor after the complete atom norm."""
    sigma = coordinates.bandwidth_sigma(layer.kernel, layer.chart, p).detach().numpy()
    amp = (
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
        for dim, size in enumerate(sizes):
            origin, period = (
                float(chart.origin[dim]),
                float(chart.geometry.periods[dim]),
            )
            width = layer.kernel.spec.tile_support.tile_widths[dim]
            tiles = round(period / width)
            count = size // tiles
            # Python branch choices are detached; continuous local centres are not.
            center_value = float(p[a, dim + 2].detach()) - origin
            turns = math.floor(center_value / period)
            owner = math.floor((center_value - turns * period) / width)
            local = p[a, dim + 2] - origin - turns * period - owner * width
            left = float(local.detach()) <= width / 2
            pair = (owner - int(left)) % tiles
            local = local + width * int(left)
            spec = layer.kernel.spec.profiles[dim].profile
            h, _ = native_tables(spec, 2 * width)
            j = max(
                0,
                min(
                    np.searchsorted(spec.width_knots, sigma[a], side="right") - 1,
                    len(h) - 2,
                ),
            )
            t = math.log(sigma[a] / spec.width_knots[j]) / math.log(
                spec.width_knots[j + 1] / spec.width_knots[j]
            )
            freq = torch.arange(-spec.modes // 2, spec.modes // 2 + 1, dtype=p.dtype)
            hj, hn = p.new_tensor(h[j]), p.new_tensor(h[j + 1])
            step = (
                float(chart.spacing[dim])
                if hasattr(chart, "coordinate_shape")
                else period / size
            )
            axis = []
            for site in range(size):
                tile, offset = divmod(site, count)
                relative = (tile - pair) % tiles
                if relative >= 2:
                    axis.append(p.new_zeros(()))
                else:
                    d = offset * step + relative * width - local
                    b = torch.cos(freq * math.pi / width * d)
                    axis.append(
                        (1 - t) * (b * hj).sum().square() + t * (b * hn).sum().square()
                    )
            axes.append(torch.stack(axis))
        atom = axes[0][:, None] * axes[1][None, :]
        result = result + amp[a] * atom / torch.linalg.vector_norm(atom).clamp_min(
            layer.kernel.spec.normalization.floor
        )
    return result


@pytest.mark.parametrize("modes", [16, 24, 32])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_full_site_values_and_all_gradients_including_one_product_floor(modes, floor):
    layer = model(modes, floor=floor)
    p = layer.atoms.p
    x = torch.randn(2, 400, dtype=p.dtype, requires_grad=True)
    target = torch.randn(2, 400, dtype=p.dtype)
    expected = x @ independent_weight(layer, p).T
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    observed = torch.autograd.grad((actual * target).sum(), (x, p))
    truth = torch.autograd.grad((expected * target).sum(), (x, p))
    for value, reference in zip(observed, truth):
        torch.testing.assert_close(value, reference, rtol=3e-10, atol=3e-11)
    torch.testing.assert_close(
        (observed[1][:, :2] * p[:, :2]).sum(-1),
        torch.zeros(len(p), dtype=p.dtype),
        rtol=0,
        atol=2e-11,
    )


def test_exact_two_tiles_midpoint_tie_seam_and_global_turn():
    layer = model()
    centers = torch.tensor(
        [[-0.3, 0.2], [9.7, 0.2], [9.700001, 0.2], [19.7, 0.2], [79.7, 0.2]],
        dtype=torch.float64,
    )
    d, mask = axis_geometry(layer.chart, centers, 0, 20.0, 400)
    assert torch.equal(mask.sum(0), torch.full((5,), 200))
    assert mask[399, 0] and mask[0, 0] and not mask[100, 0]
    assert torch.equal(mask[:, 0], mask[:, 1])  # exact midpoint chooses left
    assert mask[100, 2] and not mask[399, 2]
    torch.testing.assert_close(mask[:, 0], mask[:, 4])
    torch.testing.assert_close(d[mask[:, 0], 0], d[mask[:, 4], 4])
    assert two_tile_layout(layer.chart.spec, layer.kernel.spec) == ((4, 100, 20.0),) * 2
    with pytest.raises(ValueError):
        fourier_tile_layout(layer.chart.spec, layer.kernel.spec)


def test_checkpoint_contract_covers_template_knots_modes_and_tile_support():
    layer = model(regular=True, dtype=torch.float32)
    clone = model(regular=True, dtype=torch.float32)
    clone.load_state_dict(copy.deepcopy(layer.state_dict()))
    x = torch.randn(2, 400)
    torch.testing.assert_close(clone(x), layer(x))
    other = model(24, regular=True, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        other.load_state_dict(layer.state_dict())
    changed = replace(
        layer.kernel.spec,
        profiles=tuple(
            replace(
                binding, profile=replace(binding.profile, width_knots=(0.1, 1.0, 10.0))
            )
            for binding in layer.kernel.spec.profiles
        ),
    )
    other = CSTLinear(
        chart=layer.chart,
        kernel=changed,
        atoms=layer.atoms.p.detach().clone(),
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        other.load_state_dict(layer.state_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"modes": 17},
        {"modes": True},
        {"concentration_floor": 10},
        {"width_knots": (0.1, 0.1)},
        {"width_knots": (0.1, float("nan"))},
    ],
)
def test_invalid_template_declarations(kwargs):
    with pytest.raises(ValueError):
        KaiserTemplateSpec(**kwargs)


def test_invalid_support_bounds_and_old_family_mixing_are_rejected():
    source = model()
    with pytest.raises(ValueError):
        two_tile_layout(
            source.chart.spec,
            replace(
                source.kernel.spec,
                tile_support=FourierTwoTileSupportSpec(tile_widths=(10.0,) * 2),
            ),
        )
    with pytest.raises(ValueError):
        presets.polar_two_tile_template_product(
            profiles=(KaiserTemplateSpec(),) * 2,
            tile_widths=(10.0,) * 2,
            amplitude_max=1.0,
            w_c=0.1,
            bounds=BandwidthBounds(
                minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1
            ),
        )
    with pytest.raises(ValueError):
        replace(
            source.kernel.spec,
            profiles=(
                replace(source.kernel.spec.profiles[0], profile=KaiserSquareSpec()),
            )
            * 2,
        )
