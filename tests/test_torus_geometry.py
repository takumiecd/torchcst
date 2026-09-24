"""Toroidal site geometry and its Strip locality contract."""

import copy
import math

import pytest
import torch

from torchcst import (
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    ProductChart,
    StripChart,
    TorusGeometry,
    Triweight,
)


def _kernel(radius: float) -> DirectAmpWidth:
    return DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=radius,
        sigma_birth=radius,
        sigma_max=radius,
        w_c=0.05,
        kappa=3.0,
        profile=Triweight(radius, normalize_columns=False),
    )


def _strip(representation: str = "ambient") -> StripChart:
    return StripChart(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(LinePattern(16, spacing=2.0), GridPattern((2, 2), spacing=0.2)),
        axis=0,
        tile_pitch=25.0,
        geometry=TorusGeometry(
            3,
            major_radius=100 / (2 * math.pi),
            minor_radius=1.0,
            max_arc_step=10.0,
            representation=representation,
        ),
    ).double()


def test_torus_has_three_degrees_of_freedom_in_four_dimensions() -> None:
    geometry = TorusGeometry(3, major_radius=8.0, minor_radius=2.0).double()
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [math.pi * 8.0, 0.0, 0.0], [0.5, 0.3, -0.2]],
        dtype=torch.float64,
    )
    sites = geometry.lift_chart_coordinates(coordinates)
    assert geometry.intrinsic_dim == 3
    assert geometry.embedding_dim == geometry.center_parameter_dim == 4
    assert sites.shape == (3, 4)
    geometry.validate_points(sites)
    torch.testing.assert_close(
        sites[0], torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        geometry.squared_distance(sites[:1], sites[1:2]),
        torch.tensor([[400.0]], dtype=torch.float64),
    )


def test_torus_distance_matches_closed_form_and_gemm_identity() -> None:
    geometry = TorusGeometry(3, major_radius=8.0, minor_radius=2.0).double()
    sites = geometry.lift_chart_coordinates(
        torch.tensor([[0.2, 0.4, 0.1], [1.3, -0.2, 0.3]], dtype=torch.float64)
    )
    centers = sites.flip(0)
    squared = geometry.squared_distance(sites, centers)
    gemm = (
        sites.square().sum(-1, keepdim=True)
        + centers.square().sum(-1)[None, :]
        - 2 * sites @ centers.T
    )
    torch.testing.assert_close(squared, gemm, atol=1e-12, rtol=1e-12)
    for site_index in range(2):
        for center_index in range(2):
            site, center = sites[site_index], centers[center_index]
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
    geometry = TorusGeometry(
        2, major_radius=5.0, minor_radius=1.0, max_arc_step=0.25
    ).double()
    sites = geometry.lift_chart_coordinates(
        torch.tensor([[0.0, 0.0], [0.4, 0.2]], dtype=torch.float64)
    )
    old = sites[:1]
    displacement = torch.tensor([[0.5, 0.8, -0.4]], dtype=torch.float64)
    new = geometry.retract(old, displacement)
    geometry.validate_centers(new)
    old_angle = torch.atan2(old[:, 1], old[:, 0])
    new_angle = torch.atan2(new[:, 1], new[:, 0])
    assert bool(((new_angle - old_angle).abs() <= 0.25 / 5.0 + 1e-12).all())
    transported = geometry.transport(old, new, displacement)
    normal = geometry._normal(new)
    torch.testing.assert_close(
        (normal * transported).sum(-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )

    chart = ProductChart(
        shape=(2, 2),
        axes=(LinePattern(2, spacing=0.4), LinePattern(2, spacing=0.2)),
        geometry=geometry,
    ).double()
    profile = Triweight(1.8).double()
    center = chart.positions(torch.tensor([0]))
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    _, tangent, _ = profile.tangent_with_precision(chart, center, precision)
    jacobian = torch.autograd.functional.jacobian(
        lambda p: profile.evaluate_with_precision(chart, p, precision), center
    )
    torch.testing.assert_close(tangent[:, 0], jacobian[:, 0, 0], atol=1e-10, rtol=1e-10)


def test_intrinsic_torus_stores_three_coordinates_and_matches_ambient_distance() -> (
    None
):
    ambient = TorusGeometry(3, major_radius=8.0, minor_radius=2.0).double()
    intrinsic = TorusGeometry(
        3, major_radius=8.0, minor_radius=2.0, representation="intrinsic"
    ).double()
    sites = ambient.lift_chart_coordinates(
        torch.tensor([[0.3, 0.2, -0.1], [1.1, -0.2, 0.3]], dtype=torch.float64)
    )
    centers = intrinsic.initialize_centers(sites, 2, mode="balanced")
    assert centers.shape == (2, 3)
    assert intrinsic.center_parameter_dim == intrinsic.intrinsic_dim == 3
    assert intrinsic.embedding_dim == 4
    intrinsic.validate_centers(centers)
    decoded = intrinsic.decode_centers(centers)
    torch.testing.assert_close(decoded, sites)
    torch.testing.assert_close(
        intrinsic.squared_distance(sites, centers),
        ambient.squared_distance(sites, sites),
    )
    uniform = intrinsic.initialize_centers(sites, 32, mode="uniform")
    intrinsic.validate_centers(uniform)
    intrinsic.validate_points(intrinsic.decode_centers(uniform))


def test_intrinsic_torus_tangent_matches_autograd_and_retracts_across_seam() -> None:
    geometry = TorusGeometry(
        3,
        major_radius=5.0,
        minor_radius=1.0,
        representation="intrinsic",
        max_arc_step=0.25,
    ).double()
    chart = ProductChart(
        shape=(2, 4),
        axes=(LinePattern(2, spacing=0.4), GridPattern((2, 2), spacing=0.2)),
        geometry=geometry,
    ).double()
    profile = Triweight(1.8).double()
    precision = torch.tensor([1 / 1.8**2], dtype=torch.float64)
    for center in (
        chart.initialize_centers(1, mode="balanced"),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[0.4, 0.7, -0.5]], dtype=torch.float64),
    ):
        _, tangent, _ = profile.tangent_with_precision(chart, center, precision)
        jacobian = torch.autograd.functional.jacobian(
            lambda p: profile.evaluate_with_precision(chart, p, precision), center
        )
        torch.testing.assert_close(
            tangent[:, 0], jacobian[:, 0, 0], atol=1e-10, rtol=1e-10
        )

    old = torch.tensor([[math.pi * 5 - 0.1, 0.1, -0.2]], dtype=torch.float64)
    updated = geometry.retract(
        old, torch.tensor([[4.0, 100.0, 100.0]], dtype=torch.float64)
    )
    geometry.validate_centers(updated)
    torch.testing.assert_close(
        updated[0, 0], torch.tensor(-math.pi * 5 + 0.15, dtype=torch.float64)
    )
    assert (
        torch.linalg.vector_norm(updated[0, 1:])
        <= geometry.max_section_parameter_radius
    )
    geometry.validate_points(geometry.decode_centers(updated))


def test_torus_large_update_cannot_jump_past_arc_step() -> None:
    geometry = TorusGeometry(
        2, major_radius=5.0, minor_radius=1.0, max_arc_step=0.25
    ).double()
    old = geometry.lift_chart_coordinates(torch.zeros(1, 2, dtype=torch.float64))
    moved = geometry.retract(
        old, torch.tensor([[0.0, 100.0, 0.0]], dtype=torch.float64)
    )
    angle = torch.atan2(moved[0, 1], moved[0, 0])
    torch.testing.assert_close(angle, torch.tensor(0.05, dtype=torch.float64))
    geometry.validate_centers(moved)


@pytest.mark.parametrize("representation", ["ambient", "intrinsic"])
def test_torus_strip_support_packing_optimizer_and_checkpoint(
    representation: str,
) -> None:
    chart = _strip(representation)
    chart.validate_support(10.0)
    with pytest.raises(ValueError, match="more than two"):
        chart.validate_support(15.0)
    centers = chart.geometry.lift_chart_coordinates(
        torch.tensor([[0.5, -0.1, -0.1], [75.5, -0.1, -0.1]], dtype=torch.float64)
    )
    if representation == "intrinsic":
        centers = chart.geometry.encode_centers(centers)
    supported = chart.squared_distance(centers).reshape(4, 4, 4, 2).lt(100)
    torch.testing.assert_close(
        supported.any(dim=(1, 2)),
        torch.tensor([[True, True], [True, False], [False, False], [False, True]]),
    )
    # The first and last stations are neighbors across the circle seam.
    positions = chart.positions(torch.tensor([0, 4, 32, 48]))
    assert torch.dist(positions[0], positions[3]) < torch.dist(
        positions[0], positions[2]
    )
    model = CSTLinear(chart=chart, atoms=3, kernel=_kernel(10.0), dtype=torch.float64)
    assert model.atom_parameter_dof == 5
    packed = model.packed_weight()
    assert packed.is_contiguous()
    assert packed.shape == (4, 4, 4)
    dense = model.dense_weight().flatten()
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])

    model(torch.randn(2, 4, dtype=torch.float64)).square().mean().backward()
    CSTParameterAdam(model).step()
    chart.geometry.validate_centers(model.atoms.p[:, 2:])
    restored = CSTLinear(
        chart=_strip(representation), atoms=3, kernel=_kernel(10.0), dtype=torch.float64
    )
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored.dense_weight(), model.dense_weight())


def test_torus_support_rejects_nonlocal_wraparound() -> None:
    # The direct gap between tile 0 and tile 2 is large; the nearly closed
    # seam makes their wrapped chord too short for sigma_max=10.
    chart = StripChart(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(LinePattern(16, spacing=2.0), GridPattern((2, 2), spacing=0.2)),
        axis=0,
        tile_pitch=25.0,
        geometry=TorusGeometry(3, major_radius=82 / (2 * math.pi), minor_radius=2.0),
    )
    with pytest.raises(ValueError, match="more than two"):
        chart.validate_support(10.0)


def test_torus_circle_axis_and_single_turn_are_validated() -> None:
    with pytest.raises(ValueError, match="LinePattern"):
        StripChart(
            shape=(4, 8),
            tile_shape=(4, 2),
            axes=(GridPattern((2, 2), spacing=0.2), LinePattern(8, spacing=0.1)),
            axis=1,
            tile_pitch=2.0,
            geometry=TorusGeometry(
                3, major_radius=8.0, minor_radius=1.0, circle_axis=0
            ),
        )
    with pytest.raises(ValueError, match="less than one turn"):
        ProductChart(
            shape=(8, 2),
            axes=(LinePattern(8, spacing=1.0), LinePattern(2, spacing=0.1)),
            geometry=TorusGeometry(2, major_radius=1.0, minor_radius=0.2),
        )
    chart = StripChart(
        shape=(4, 8),
        tile_shape=(4, 2),
        axes=(GridPattern((2, 2), spacing=0.2), LinePattern(8, spacing=0.1)),
        axis=1,
        tile_pitch=2.0,
        geometry=TorusGeometry(3, major_radius=8.0, minor_radius=1.0, circle_axis=2),
    )
    chart.geometry.validate_points(chart.positions(torch.arange(chart.features)))
