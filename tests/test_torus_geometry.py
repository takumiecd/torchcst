from __future__ import annotations

from torchcst import BandwidthBounds, CSTOptimizer
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.profiles import execution as _profile

"Toroidal site geometry and its Strip locality contract."
import copy
import math

import pytest
import torch
from kernel_cases import direct_state, triweight_state

from torchcst import CSTLinear
from torchcst.charts import ChartState


def make_radial_state(radius: float) -> direct_state:
    return direct_state(
        amplitude_max=1.0,
        w_c=0.05,
        kappa=3.0,
        profile=triweight_state(radius, normalize_columns=False),
        input_bounds=BandwidthBounds(
            minimum=radius, maximum=radius, birth=radius, upper_floor=radius
        ),
        composition="radial",
    )


def _strip(representation: str = "ambient") -> ChartState:
    return _construction.strip(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(
            _construction.line_pattern(16, spacing=2.0),
            _construction.grid_pattern((2, 2), spacing=0.2),
        ),
        axis=0,
        tile_pitch=25.0,
        geometry=_construction.torus(
            3,
            major_radius=100 / (2 * math.pi),
            minor_radius=1.0,
            max_arc_step=10.0,
            representation=representation,
        ),
    ).double()


def test_torus_has_three_degrees_of_freedom_in_four_dimensions() -> None:
    geometry = _construction.torus(3, major_radius=8.0, minor_radius=2.0).double()
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [math.pi * 8.0, 0.0, 0.0], [0.5, 0.3, -0.2]],
        dtype=torch.float64,
    )
    sites = _geometry.lift_chart_coordinates(geometry, coordinates)
    assert geometry.intrinsic_dim == 3
    assert geometry.embedding_dim == geometry.center_parameter_dim == 4
    assert sites.shape == (3, 4)
    _geometry.validate_points(geometry, sites)
    torch.testing.assert_close(
        sites[0], torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        _geometry.squared_distance(geometry, sites[:1], sites[1:2]),
        torch.tensor([[400.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("size", [2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_torus_accepts_embedded_sites_but_rejects_off_surface(
    size: int, dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("CUDA required")
    chart = _construction.strip(
        shape=(size, size),
        tile_shape=(16, size),
        axes=(
            _construction.line_pattern(size, spacing=0.1),
            _construction.grid_pattern((size // 16, 16), spacing=0.05),
        ),
        axis=0,
        tile_pitch=4.1,
        geometry=_construction.torus(
            3,
            major_radius=size // 16 * 4.1 / (2 * math.pi),
            minor_radius=0.4,
            representation="intrinsic",
        ),
    ).to(device=device, dtype=dtype)
    sites = _charts.positions(
        chart,
        torch.linspace(0, size * size - 1, size * 2, device=device, dtype=torch.float64)
        .round()
        .long(),
    )
    _geometry.validate_points(chart.geometry, sites)
    centers = _charts.initialize_centers(chart, size * 2, mode="balanced")
    _geometry.validate_centers(chart.geometry, centers)
    bad = sites.new_tensor([[float(chart.geometry.major_radius) + 0.41, 0, 0, 0]])
    with pytest.raises(ValueError, match="torus surface"):
        _geometry.validate_points(chart.geometry, bad)


def test_torus_distance_matches_closed_form_and_gemm_identity() -> None:
    geometry = _construction.torus(3, major_radius=8.0, minor_radius=2.0).double()
    sites = _geometry.lift_chart_coordinates(
        geometry, torch.tensor([[0.2, 0.4, 0.1], [1.3, -0.2, 0.3]], dtype=torch.float64)
    )
    centers = sites.flip(0)
    squared = _geometry.squared_distance(geometry, sites, centers)
    gemm = (
        sites.square().sum(-1, keepdim=True)
        + centers.square().sum(-1)[None, :]
        - 2 * sites @ centers.T
    )
    torch.testing.assert_close(squared, gemm, atol=1e-12, rtol=1e-12)
    for site_index in range(2):
        for center_index in range(2):
            site, center = (sites[site_index], centers[center_index])
            theta = torch.atan2(site[1], site[0])
            phi = torch.atan2(center[1], center[0])
            a = torch.linalg.vector_norm(site[:2])
            b = torch.linalg.vector_norm(center[:2])
            q = torch.cat(((a - 8).reshape(1), site[2:])) / 2
            q_prime = torch.cat(((b - 8).reshape(1), center[2:])) / 2
            expected = (
                4 * a * b * torch.sin((theta - phi) / 2).square()
                + 4 * (q - q_prime).square().sum()
            )
            torch.testing.assert_close(squared[site_index, center_index], expected)


def test_torus_retraction_transport_and_compact_tangent() -> None:
    geometry = _construction.torus(
        2, major_radius=5.0, minor_radius=1.0, max_arc_step=0.25
    ).double()
    sites = _geometry.lift_chart_coordinates(
        geometry, torch.tensor([[0.0, 0.0], [0.4, 0.2]], dtype=torch.float64)
    )
    old = sites[:1]
    displacement = torch.tensor([[0.5, 0.8, -0.4]], dtype=torch.float64)
    new = _geometry.retract(geometry, old, displacement)
    _geometry.validate_centers(geometry, new)
    old_angle = torch.atan2(old[:, 1], old[:, 0])
    new_angle = torch.atan2(new[:, 1], new[:, 0])
    assert bool(((new_angle - old_angle).abs() <= 0.25 / 5.0 + 1e-12).all())
    transported = _geometry.transport(geometry, old, new, displacement)
    radial = torch.linalg.vector_norm(new[..., :2], dim=-1, keepdim=True)
    normal = torch.cat(
        (
            ((radial - geometry.major_radius) / geometry.minor_radius)
            * new[..., :2]
            / radial,
            new[..., 2:] / geometry.minor_radius,
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        (normal * transported).sum(-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    chart = _construction.product(
        shape=(2, 2),
        axes=(
            _construction.line_pattern(2, spacing=0.4),
            _construction.line_pattern(2, spacing=0.2),
        ),
        geometry=geometry,
    ).double()
    profile = triweight_state(1.8).double()
    center = _charts.positions(chart, torch.tensor([0]))
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    _, tangent, _ = _profile.tangent_with_precision(profile, chart, center, precision)
    jacobian = torch.autograd.functional.jacobian(
        lambda p: _profile.evaluate_with_precision(profile, chart, p, precision), center
    )
    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0], atol=1e-10, rtol=1e-10)


def test_intrinsic_torus_stores_three_coordinates_and_matches_ambient_distance() -> (
    None
):
    ambient = _construction.torus(3, major_radius=8.0, minor_radius=2.0).double()
    intrinsic = _construction.torus(
        3, major_radius=8.0, minor_radius=2.0, representation="intrinsic"
    ).double()
    sites = _geometry.lift_chart_coordinates(
        ambient, torch.tensor([[0.3, 0.2, -0.1], [1.1, -0.2, 0.3]], dtype=torch.float64)
    )
    centers = _geometry.initialize_centers(intrinsic, sites, 2, mode="balanced")
    assert centers.shape == (2, 3)
    assert intrinsic.center_parameter_dim == intrinsic.intrinsic_dim == 3
    assert intrinsic.embedding_dim == 4
    _geometry.validate_centers(intrinsic, centers)
    decoded = _geometry.decode_centers(intrinsic, centers)
    torch.testing.assert_close(decoded, sites)
    torch.testing.assert_close(
        _geometry.squared_distance(intrinsic, sites, centers),
        _geometry.squared_distance(ambient, sites, sites),
    )
    uniform = _geometry.initialize_centers(intrinsic, sites, 32, mode="uniform")
    _geometry.validate_centers(intrinsic, uniform)
    _geometry.validate_points(intrinsic, _geometry.decode_centers(intrinsic, uniform))


def test_intrinsic_torus_tangent_matches_autograd_and_retracts_across_seam() -> None:
    geometry = _construction.torus(
        3,
        major_radius=5.0,
        minor_radius=1.0,
        representation="intrinsic",
        max_arc_step=0.25,
    ).double()
    chart = _construction.product(
        shape=(2, 4),
        axes=(
            _construction.line_pattern(2, spacing=0.4),
            _construction.grid_pattern((2, 2), spacing=0.2),
        ),
        geometry=geometry,
    ).double()
    profile = triweight_state(1.8).double()
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    for center in (
        _charts.initialize_centers(chart, 1, mode="balanced"),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[0.4, 0.7, -0.5]], dtype=torch.float64),
    ):
        _, tangent, _ = _profile.tangent_with_precision(
            profile, chart, center, precision
        )
        jacobian = torch.autograd.functional.jacobian(
            lambda p: _profile.evaluate_with_precision(profile, chart, p, precision),
            center,
        )
        torch.testing.assert_close(
            tangent[:, 0], jacobian[:, 0, 0], atol=1e-10, rtol=1e-10
        )
    old = torch.tensor([[math.pi * 5 - 0.1, 0.1, -0.2]], dtype=torch.float64)
    updated = _geometry.retract(
        geometry, old, torch.tensor([[4.0, 100.0, 100.0]], dtype=torch.float64)
    )
    _geometry.validate_centers(geometry, updated)
    torch.testing.assert_close(
        updated[0, 0], torch.tensor(-math.pi * 5 + 0.15, dtype=torch.float64)
    )
    assert torch.linalg.vector_norm(
        updated[0, 1:]
    ) <= _geometry.max_section_parameter_radius(geometry)
    _geometry.validate_points(geometry, _geometry.decode_centers(geometry, updated))


def test_torus_large_update_cannot_jump_past_arc_step() -> None:
    geometry = _construction.torus(
        2, major_radius=5.0, minor_radius=1.0, max_arc_step=0.25
    ).double()
    old = _geometry.lift_chart_coordinates(
        geometry, torch.zeros(1, 2, dtype=torch.float64)
    )
    moved = _geometry.retract(
        geometry, old, torch.tensor([[0.0, 100.0, 0.0]], dtype=torch.float64)
    )
    angle = torch.atan2(moved[0, 1], moved[0, 0])
    torch.testing.assert_close(angle, torch.tensor(0.05, dtype=torch.float64))
    _geometry.validate_centers(geometry, moved)


@pytest.mark.parametrize("representation", ["ambient", "intrinsic"])
def test_torus_strip_support_packing_optimizer_and_checkpoint(
    representation: str,
) -> None:
    chart = _strip(representation)
    _charts.validate_support(chart, 10.0)
    with pytest.raises(ValueError, match="more than two"):
        _charts.validate_support(chart, 15.0)
    centers = _geometry.lift_chart_coordinates(
        chart.geometry,
        torch.tensor([[0.5, -0.1, -0.1], [75.5, -0.1, -0.1]], dtype=torch.float64),
    )
    if representation == "intrinsic":
        centers = _geometry.encode_centers(chart.geometry, centers)
    supported = _charts.squared_distance(chart, centers).reshape(4, 4, 4, 2).lt(100)
    torch.testing.assert_close(
        supported.any(dim=(1, 2)),
        torch.tensor([[True, True], [True, False], [False, False], [False, True]]),
    )
    positions = _charts.positions(chart, torch.tensor([0, 4, 32, 48]))
    assert torch.dist(positions[0], positions[3]) < torch.dist(
        positions[0], positions[2]
    )
    model = CSTLinear(
        chart=chart,
        atoms=3,
        kernel=make_radial_state(10.0).declaration(),
        dtype=torch.float64,
    )
    assert model.atom_parameter_dof == 5
    packed = model.packed_weight()
    assert packed.is_contiguous()
    assert packed.shape == (4, 4, 4)
    dense = model.dense_weight().flatten()
    for station in range(chart.tile_count):
        logical, local = _charts.tile_indices(chart, station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
    model(torch.randn(2, 4, dtype=torch.float64)).square().mean().backward()
    CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    ).step()
    _geometry.validate_centers(chart.geometry, model.atoms.p[:, 2:])
    restored = CSTLinear(
        chart=_strip(representation),
        atoms=3,
        kernel=make_radial_state(10.0).declaration(),
        dtype=torch.float64,
    )
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored.dense_weight(), model.dense_weight())


def test_torus_support_rejects_nonlocal_wraparound() -> None:
    chart = _construction.strip(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(
            _construction.line_pattern(16, spacing=2.0),
            _construction.grid_pattern((2, 2), spacing=0.2),
        ),
        axis=0,
        tile_pitch=25.0,
        geometry=_construction.torus(
            3, major_radius=82 / (2 * math.pi), minor_radius=2.0
        ),
    )
    with pytest.raises(ValueError, match="more than two"):
        _charts.validate_support(chart, 10.0)


def test_torus_circle_axis_and_single_turn_are_validated() -> None:
    with pytest.raises(ValueError, match="line axis"):
        _construction.strip(
            shape=(4, 8),
            tile_shape=(4, 2),
            axes=(
                _construction.grid_pattern((2, 2), spacing=0.2),
                _construction.line_pattern(8, spacing=0.1),
            ),
            axis=1,
            tile_pitch=2.0,
            geometry=_construction.torus(
                3, major_radius=8.0, minor_radius=1.0, circle_axis=0
            ),
        )
    with pytest.raises(ValueError, match="less than one turn"):
        _construction.product(
            shape=(8, 2),
            axes=(
                _construction.line_pattern(8, spacing=1.0),
                _construction.line_pattern(2, spacing=0.1),
            ),
            geometry=_construction.torus(2, major_radius=1.0, minor_radius=0.2),
        )
    chart = _construction.strip(
        shape=(4, 8),
        tile_shape=(4, 2),
        axes=(
            _construction.grid_pattern((2, 2), spacing=0.2),
            _construction.line_pattern(8, spacing=0.1),
        ),
        axis=1,
        tile_pitch=2.0,
        geometry=_construction.torus(
            3, major_radius=8.0, minor_radius=1.0, circle_axis=2
        ),
    )
    _geometry.validate_points(
        chart.geometry, _charts.positions(chart, torch.arange(chart.features))
    )
