"""Radius-native probabilities and all physical gradients against quadrature."""

import math
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pytest
import torch

from torchcst import (
    ChebyshevRadiusSpec,
    CSTLinear,
    RadiusBounds,
    RadiusIntegratedProfileBinding,
    chart_presets,
    presets,
)
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses
from torchcst._backends.torch.profiles.radius_fourier import (
    density_coefficients,
    support_eps,
)
from torchcst.kernels.radius_cell_mass import RadiusCellMassGrid, radius_cell_mass_axes


def axis(n=16, *, spacing=1.0, origin=0.0, modes=8, merge=2):
    return RadiusCellMassGrid(
        n,
        spacing,
        origin,
        RadiusIntegratedProfileBinding(
            profile=ChebyshevRadiusSpec(modes=modes), merge_factor=merge
        ),
    )


@lru_cache(None)
def quadrature_rule():
    return np.polynomial.legendre.leggauss(192)


def integral(radius, period, modes, left, right):
    """Independent raw T_M polynomial quadrature, without FFT coefficients."""
    nodes, weights = quadrature_rule()
    x = (right - left) / 2 * nodes + (right + left) / 2
    beta = 1 / math.cos(math.pi * radius / period)
    coefficients = np.zeros(modes + 1)
    coefficients[-1] = 1
    value = np.polynomial.chebyshev.chebval(
        beta * np.cos(math.pi * x / period), coefficients
    )
    value /= math.cosh(modes * math.acosh(beta))
    return float((right - left) / 2 * np.dot(weights, value * value))


def oracle_mass(grid, radius, center):
    tile = grid.binding.base_tile_sites
    while radius > tile * grid.spacing:
        tile *= grid.binding.merge_factor
    width = tile * grid.spacing
    count = math.ceil(2 * radius / width) + 1
    period = count * width
    center = (center - grid.origin) % grid.period
    first = math.floor((center + grid.spacing / 2) / width - (count - 1) / 2) * tile
    normalization = integral(
        radius, period, grid.binding.profile.modes, -period / 2, period / 2
    )
    result = np.zeros(grid.site_count)
    for site in range(first, first + count * tile):
        left = (site - 0.5) * grid.spacing - center
        right = left + grid.spacing
        result[site % grid.site_count] += (
            integral(radius, period, grid.binding.profile.modes, left, right)
            / normalization
        )
    return result


@pytest.mark.parametrize("modes", [1, 3, 5, 8, 10, 16])
@pytest.mark.parametrize(
    "n,spacing,origin,merge",
    [(16, 1.0, 0.0, 2), (16, 0.125, -0.7, 2), (27, 0.37, 1.2, 3)],
)
def test_cells_match_independent_quadrature(modes, n, spacing, origin, merge):
    grid = axis(n, spacing=spacing, origin=origin, modes=modes, merge=merge)
    radii = (
        torch.tensor([0.5, 0.63, 1.0, 1.3, 2.7, n / 2], dtype=torch.float64) * spacing
    )
    centers = (
        torch.tensor([0.0, 0.25, 0.5, -0.31, n - 0.27, n + 1.1], dtype=torch.float64)
        * spacing
        + origin
    )
    actual = axis_masses(grid, radii, centers)
    expected = np.stack(
        [oracle_mass(grid, float(r), float(c)) for r, c in zip(radii, centers)]
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=3e-12, rtol=3e-12
    )
    torch.testing.assert_close(
        actual.sum(1), torch.ones(6, dtype=torch.float64), atol=3e-12, rtol=0
    )
    assert float(actual.min()) > -3e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("modes", [3, 8])
@pytest.mark.parametrize("n,merge", [(16, 2), (27, 3), (128, 2)])
def test_decimal_hierarchy_thresholds_and_neighbors_match_encoded_fp64_and_quadrature(
    dtype, modes, n, merge
):
    grid = axis(n, spacing=0.37, origin=-1.2, modes=modes, merge=merge)
    radii = []
    tile = grid.binding.base_tile_sites
    while tile * grid.spacing <= grid.period / 2:
        boundary = torch.tensor(tile * grid.spacing, dtype=dtype)
        for value in (
            torch.nextafter(boundary, torch.full_like(boundary, -math.inf)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, math.inf)),
        ):
            if 0 < value.item() <= grid.period / 2:
                radii.append(value)
        tile *= merge
    radius = torch.stack(radii)
    centers = torch.linspace(-1.43, grid.period + 0.17, len(radius), dtype=dtype)
    actual = axis_masses(grid, radius, centers)
    # Conversion retains exactly the encoded parameters; only input dtype differs.
    same_values = axis_masses(grid, radius.double(), centers.double())
    torch.testing.assert_close(actual, same_values, atol=0, rtol=0)
    expected = np.stack(
        [oracle_mass(grid, float(r), float(c)) for r, c in zip(radius, centers)]
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=3e-12, rtol=3e-12
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [16, 128])
def test_decimal_chart_cap_uses_physical_threshold_not_fp32_rounded_threshold(dtype, n):
    grid = axis(n, spacing=0.37, modes=3)
    cap = torch.tensor(grid.period / 2, dtype=dtype)
    if cap.item() > grid.period / 2:
        cap = torch.nextafter(cap, torch.full_like(cap, -math.inf))
    above = torch.nextafter(cap, torch.full_like(cap, math.inf))
    assert cap.item() <= grid.period / 2 < above.item()
    actual = axis_masses(grid, cap[None], torch.zeros(1, dtype=dtype))
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0, dtype=torch.float64))
    for radius in (above[None], above[None].double()):
        with pytest.raises(ValueError, match="chart period/2"):
            axis_masses(grid, radius, torch.zeros_like(radius))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("modes", [3, 8])
def test_public_decimal_boundary_y_dx_and_all_atom_gradients_against_quadrature(
    dtype, modes
):
    grid = axis(16, spacing=0.37, origin=-1.2, modes=modes)
    chart = chart_presets.regular_grid(
        ((16,),), spacing=grid.spacing, origin=grid.origin, geometry="flat_torus"
    )
    boundary = torch.tensor(grid.spacing, dtype=dtype)
    radii = torch.stack(
        [
            torch.nextafter(boundary, torch.full_like(boundary, -math.inf)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, math.inf)),
        ]
    )
    p = torch.tensor(
        [[0.3, 0.0, -0.23, 3.31], [-0.2, 0.0, 4.57, 0.23], [0.4, 0.0, 1.17, -0.71]],
        dtype=dtype,
    )
    p[:, 1] = radii
    model = CSTLinear(
        chart,
        chart,
        atoms=p,
        kernel=presets.radius_cell_mass(profile=ChebyshevRadiusSpec(modes=modes)),
        dtype=dtype,
        backend="factored",
    )
    # Public chart buffers follow model dtype, so use their resulting physical
    # declaration in the scalar oracle rather than the pre-conversion decimals.
    declaration = model.declaration()
    grid = radius_cell_mass_axes(declaration.charts, declaration.kernel)[0]
    gen = torch.Generator().manual_seed(742)
    x = torch.randn(3, 16, dtype=dtype, generator=gen, requires_grad=True)
    dy = torch.randn(3, 16, dtype=dtype, generator=gen)
    actual = model(x)
    actual.backward(dy)
    anchors = model.atoms.p.detach().double().numpy().copy()

    def fixed_route_mass(radius, center, anchor_radius, anchor_center):
        # At a hierarchy boundary the contract differentiates the selected smooth
        # branch. Freeze only integer routing for quadrature finite differences.
        tile = grid.binding.base_tile_sites
        while anchor_radius > tile * grid.spacing:
            tile *= grid.binding.merge_factor
        width = tile * grid.spacing
        count = math.ceil(2 * anchor_radius / width) + 1
        period = count * width
        routed_center = (anchor_center - grid.origin) % grid.period
        first = (
            math.floor((routed_center + grid.spacing / 2) / width - (count - 1) / 2)
            * tile
        )
        center = (center - grid.origin) % grid.period
        normalization = integral(radius, period, modes, -period / 2, period / 2)
        mass = np.zeros(grid.site_count)
        for site in range(first, first + count * tile):
            left = (site - 0.5) * grid.spacing - center
            mass[site % grid.site_count] += (
                integral(radius, period, modes, left, left + grid.spacing)
                / normalization
            )
        return mass

    def independent_weight(parameters):
        weight = np.zeros((16, 16))
        for row, anchor in zip(parameters, anchors):
            amplitude, radius, output_center, input_center = row
            outputs = fixed_route_mass(radius, output_center, anchor[1], anchor[2])
            inputs = fixed_route_mass(radius, input_center, anchor[1], anchor[3])
            weight += amplitude * np.outer(outputs, inputs)
        return weight

    weight = independent_weight(anchors)
    # Also verify the frozen branch at its anchor with the ordinary scalar oracle.
    ordinary = sum(
        amplitude
        * np.outer(
            oracle_mass(grid, radius, output_center),
            oracle_mass(grid, radius, input_center),
        )
        for amplitude, radius, output_center, input_center in anchors
    )
    np.testing.assert_allclose(weight, ordinary, atol=3e-12, rtol=3e-12)
    tolerance = (
        dict(rtol=3e-6, atol=2e-7)
        if dtype == torch.float32
        else dict(rtol=2e-11, atol=2e-12)
    )
    np.testing.assert_allclose(
        actual.detach().numpy(), x.detach().numpy() @ weight.T, **tolerance
    )
    np.testing.assert_allclose(x.grad.numpy(), dy.numpy() @ weight, **tolerance)
    step = 1e-5
    numerical = np.zeros_like(anchors)
    for atom in range(len(anchors)):
        for column in range(4):
            plus, minus = anchors.copy(), anchors.copy()
            plus[atom, column] += step
            minus[atom, column] -= step
            numerical[atom, column] = np.sum(
                (
                    x.detach().numpy()
                    @ (independent_weight(plus) - independent_weight(minus)).T
                )
                * dy.numpy()
            ) / (2 * step)
    np.testing.assert_allclose(
        model.atoms.p.grad.numpy(),
        numerical,
        rtol=3e-6 if dtype == torch.float32 else 2e-6,
        atol=2e-7 if dtype == torch.float32 else 2e-8,
    )


def test_plus_one_padding_and_chart_folding_are_not_dropped():
    grid = axis(16)
    radius = torch.tensor([8.0], dtype=torch.float64)
    center = torch.tensor([0.13], dtype=torch.float64)
    # T=8, K=ceil(16/8)+1=3, so P=24 > Q=16.
    actual = axis_masses(grid, radius, center)[0]
    expected = oracle_mass(grid, 8.0, 0.13)
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=3e-12, rtol=3e-12
    )
    assert abs(float(actual.sum()) - 1) < 3e-12


@pytest.mark.parametrize("modes", [3, 8])
def test_center_and_radius_gradients_in_smooth_routing_branch(modes):
    grid = axis(modes=modes)
    radius = torch.tensor([0.63, 1.3, 2.7], dtype=torch.float64, requires_grad=True)
    center = torch.tensor([0.17, 3.31, 15.77], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda r, c: axis_masses(grid, r, c),
        (radius, center),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-5,
    )


def test_support_eps_is_derived_for_each_m_not_a_fixed_requirement():
    leaks = []
    for modes in [3, 5, 8, 10]:
        profile = ChebyshevRadiusSpec(modes=modes)
        radius = torch.tensor([0.5], dtype=torch.float64)
        # User's corrected rule gives P=(ceil(1)+1)*1=2, not 1 or 3.
        actual = support_eps(profile, radius, 2.0)[0]
        expected = 1 - integral(0.5, 2.0, modes, -0.5, 0.5) / integral(
            0.5, 2.0, modes, -1.0, 1.0
        )
        assert abs(float(actual) - expected) < 3e-12
        leaks.append(float(actual))
    assert all(a > b for a, b in pairwise(leaks))


def physical_layer(*, backend="factored", dtype=torch.float64):
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    p = torch.tensor([[0.3, 0.63, 2.17, 3.31], [-0.2, 1.3, 15.77, 0.23]], dtype=dtype)
    kernel = presets.radius_cell_mass(profile=ChebyshevRadiusSpec(modes=5))
    return CSTLinear(chart, chart, atoms=p, kernel=kernel, dtype=dtype, backend=backend)


def independent_y(p, x):
    grid = axis(modes=5)
    weight = np.zeros((16, 16))
    for amplitude, radius, output_center, input_center in p:
        inputs = oracle_mass(grid, radius, input_center)
        outputs = oracle_mass(grid, radius, output_center)
        weight += amplitude * outputs[:, None] * inputs[None, :]
    return x @ weight.T


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_linear_y_dx_and_all_atom_columns_against_quadrature(dtype):
    model = physical_layer(dtype=dtype)
    rng = torch.Generator().manual_seed(731)
    x = torch.randn(3, 16, dtype=dtype, generator=rng, requires_grad=True)
    dy = torch.randn(3, 16, dtype=dtype, generator=rng)
    actual = model(x)
    actual.backward(dy)
    p = model.atoms.p.detach().double().numpy().copy()
    rtol, atol = (3e-6, 2e-7) if dtype == torch.float32 else (2e-11, 2e-12)
    expected = independent_y(p, x.detach().numpy())
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=rtol, atol=atol)
    weight = independent_y(p, np.eye(16)).T
    np.testing.assert_allclose(
        x.grad.numpy(), dy.numpy() @ weight, rtol=rtol, atol=atol
    )
    step = 1e-5
    numerical = np.zeros_like(p)
    for a in range(len(p)):
        for column in range(4):
            plus, minus = p.copy(), p.copy()
            plus[a, column] += step
            minus[a, column] -= step
            numerical[a, column] = np.sum(
                (
                    independent_y(plus, x.detach().numpy())
                    - independent_y(minus, x.detach().numpy())
                )
                * dy.numpy()
            ) / (2 * step)
    np.testing.assert_allclose(
        model.atoms.p.grad.numpy(),
        numerical,
        rtol=3e-6 if dtype == torch.float32 else 2e-6,
        atol=2e-7 if dtype == torch.float32 else 2e-8,
    )


def test_factored_dense_and_retained_backward_agree():
    model = physical_layer()
    x = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    y = model(x)
    torch.testing.assert_close(y, x @ model.dense_weight().T)
    g1 = torch.autograd.grad(y.sum(), (x, model.atoms.p), retain_graph=True)
    g2 = torch.autograd.grad(y.sum(), (x, model.atoms.p))
    for first, second in zip(g1, g2):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_polar_task_radius_is_detached_but_radius_query_is_live():
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    bounds = RadiusBounds(minimum=0.5, birth=1.0, maximum=4.0, upper_floor=1.0)
    kernel = presets.polar_radius_cell_mass(
        amplitude_max=1.0,
        input_radius_bounds=bounds,
        w_c=0.3,
        profile=ChebyshevRadiusSpec(modes=5),
    )
    p = torch.tensor(
        [[1.0, 0.5, 2.17, 3.31], [0.5, 1.0, 15.77, 0.23]], dtype=torch.float64
    )
    model = CSTLinear(
        chart, chart, atoms=p, kernel=kernel, dtype=torch.float64, backend="factored"
    )
    radii = model.support_radii()
    assert all(r.requires_grad for r in radii)
    model(torch.randn(3, 16, dtype=torch.float64)).square().sum().backward()
    # Task gradients are tangent to Polar (s,t); no task-width/radial path.
    dot = (model.atoms.p.grad[:, :2] * model.atoms.p.detach()[:, :2]).sum(1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=2e-12, rtol=0)


@pytest.mark.parametrize("polar", [False, True])
def test_empty_atoms_and_batch(polar):
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    kernel = (
        presets.polar_radius_cell_mass(
            amplitude_max=1.0,
            input_radius_bounds=RadiusBounds(
                minimum=0.5, birth=1.0, maximum=4.0, upper_floor=1.0
            ),
            w_c=0.3,
        )
        if polar
        else presets.radius_cell_mass()
    )
    # The public CSTLinear contract requires a positive atom count.
    model = CSTLinear(
        chart, chart, atoms=2, kernel=kernel, dtype=torch.float64, backend="factored"
    )
    x = torch.randn(0, 16, dtype=torch.float64, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert y.shape == (0, 16)
    assert torch.equal(model.atoms.p.grad, torch.zeros_like(model.atoms.p))
    radius = torch.empty(0, dtype=torch.float64, requires_grad=True)
    center = torch.empty(0, dtype=torch.float64, requires_grad=True)
    masses = axis_masses(axis(), radius, center)
    assert masses.shape == (0, 16)
    masses.sum().backward()
    assert radius.grad.shape == center.grad.shape == (0,)


def test_invalid_runtime_values_and_hierarchy_are_rejected():
    grid = axis()
    for r in [0.0, -1.0, float("nan"), float("inf"), 8.1]:
        with pytest.raises(ValueError):
            axis_masses(grid, torch.tensor([r]), torch.tensor([0.0]))
    chart = chart_presets.regular_grid(((16,),), spacing=1.0, geometry="flat_torus")
    with pytest.raises(ValueError, match="power of merge_factor"):
        CSTLinear(
            chart, chart, atoms=1, kernel=presets.radius_cell_mass(merge_factor=3)
        )
    with pytest.raises(ValueError, match="base_tile_sites must divide"):
        CSTLinear(
            chart,
            chart,
            atoms=1,
            kernel=presets.radius_cell_mass(input_base_tile_sites=3),
        )


def test_empty_density_and_high_or_odd_modes_are_finite():
    empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
    assert density_coefficients(ChebyshevRadiusSpec(modes=3), empty, 2.0).shape == (
        0,
        7,
    )
    for modes in [1, 7, 64, 512]:
        value = density_coefficients(
            ChebyshevRadiusSpec(modes=modes),
            torch.tensor([0.5], dtype=torch.float64),
            2.0,
        )
        assert torch.isfinite(value).all()
        assert abs(float(value[0, modes]) * 2 - 1) < 1e-12
