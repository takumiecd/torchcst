"""Correctness gates for the exploratory Strip + Torus execution path."""

import math

import pytest
import torch

from torchcst import (
    CSTLinear,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    StripChart,
    TorusGeometry,
    Triweight,
)
from torchcst.nn._layout import initial_layout, plan_repack
from torchcst.nn._strip_torus import (
    owners_from_support,
    route_atoms,
    support_mask,
    tiled_linear,
)


def _model() -> CSTLinear:
    chart = StripChart(
        shape=(16, 4),
        tile_shape=(4, 4),
        axes=(LinePattern(16, spacing=2.0), GridPattern((2, 2), spacing=0.2)),
        axis=0,
        tile_pitch=25.0,
        geometry=TorusGeometry(
            3,
            major_radius=100 / (2 * math.pi),
            minor_radius=1.0,
            representation="intrinsic",
        ),
    )
    kernel = DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=10.0,
        sigma_birth=10.0,
        sigma_max=10.0,
        w_c=0.05,
        profile=Triweight(10.0, normalize_columns=False),
        checkpoint_blocks=False,
    )
    return CSTLinear(chart=chart, atoms=9, kernel=kernel, dtype=torch.float64)


def test_tiled_forward_and_gradients_match_dense_oracle_across_seam() -> None:
    torch.manual_seed(1)
    model = _model()
    with torch.no_grad():
        # This atom reaches the last station and station zero across the seam.
        point = torch.tensor([[75.5, 0.0, 0.0]], dtype=torch.float64)
        center = model.chart.geometry.lift_chart_coordinates(point)
        model.atoms.p[0, 2:] = model.chart.geometry.encode_centers(center)[0]
        model.atoms.p[0, 0] = 0.4

    touched = support_mask(model.chart, model.kernel, model.atoms.p)
    assert touched[:, 0].tolist() == [True, False, False, True]
    assert bool((touched.sum(0) <= 2).all())
    owners = owners_from_support(model.chart, model.atoms.p, touched)
    layout = initial_layout(owners, model.chart.tile_count)
    inputs = torch.randn(2, 3, model.in_features, dtype=torch.float64)

    expected = model(inputs)
    actual = tiled_linear(model.chart, model.kernel, inputs, model.atoms.p, layout)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    expected.square().sum().backward()
    expected_grad = model.atoms.p.grad.clone()
    model.atoms.p.grad = None
    actual.square().sum().backward()
    torch.testing.assert_close(
        model.atoms.p.grad, expected_grad, atol=1e-10, rtol=1e-10
    )


def test_three_phase_repack_handles_both_directions_across_seam() -> None:
    old_offsets = torch.tensor([0, 2, 4, 6, 8])
    destination = torch.tensor([3, 0, 1, 1, 2, 3, 0, 3])
    layout = plan_repack(old_offsets, destination)
    assert layout.offsets.tolist() == [0, 2, 4, 5, 8]
    assert destination[layout.order].tolist() == [0, 0, 1, 1, 2, 3, 3, 3]
    new_block_at_old_slot = torch.bucketize(
        torch.arange(8), layout.offsets[1:-1], right=True
    )
    torch.testing.assert_close(
        new_block_at_old_slot - layout.adjusted_move, destination
    )
    torch.testing.assert_close(layout.pack(torch.arange(8)), layout.order)


def test_repacked_atoms_feed_tiled_forward_after_seam_crossing() -> None:
    model = _model()
    with torch.no_grad():
        seam = torch.tensor([[75.5, 0.0, 0.0]], dtype=torch.float64)
        point = model.chart.geometry.lift_chart_coordinates(seam)
        model.atoms.p[0, 2:] = model.chart.geometry.encode_centers(point)[0]
        first_support = support_mask(model.chart, model.kernel, model.atoms.p)
        first_owners = owners_from_support(model.chart, model.atoms.p, first_support)
        first_layout = initial_layout(first_owners, model.chart.tile_count)
        old_packed = first_layout.pack(model.atoms.p)

        moved = old_packed.clone()
        packed_slot = int((first_layout.order == 0).nonzero()[0])
        next_point = torch.tensor([[62.0, 0.0, 0.0]], dtype=torch.float64)
        next_center = model.chart.geometry.lift_chart_coordinates(next_point)
        moved[packed_slot, 2:] = model.chart.geometry.encode_centers(next_center)[0]
        new_support = support_mask(model.chart, model.kernel, moved)
        new_owners = owners_from_support(model.chart, moved, new_support)
        assert int(new_owners[packed_slot]) == model.chart.tile_count - 1
        second_layout = plan_repack(first_layout.offsets, new_owners)
        inputs = torch.randn(2, model.in_features, dtype=torch.float64)
        actual = tiled_linear(
            model.chart,
            model.kernel,
            inputs,
            moved,
            second_layout,
        )
        expected = inputs @ model.kernel.weight(model.chart, moved).T
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_repack_accepts_multi_station_jumps() -> None:
    old_offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    destination = torch.tensor([4, 4, 1, 5, 0, 3, 0, 0, 2, 2, 5, 0])
    layout = plan_repack(old_offsets, destination)
    assert layout.offsets.tolist() == [0, 4, 5, 7, 8, 10, 12]
    assert layout.order.tolist() == [4, 6, 7, 11, 2, 8, 9, 5, 0, 1, 3, 10]
    new_block_at_old_slot = torch.bucketize(
        torch.arange(12), layout.offsets[1:-1], right=True
    )
    torch.testing.assert_close(
        new_block_at_old_slot - layout.adjusted_move, destination
    )


def test_repack_rejects_a_destination_outside_the_station_range() -> None:
    with pytest.raises(ValueError, match="destination outside"):
        plan_repack(torch.tensor([0, 1, 2]), torch.tensor([2, 1]))


@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
def test_circle_routing_owns_a_supported_station_and_covers_every_site(representation):
    torch.manual_seed(57)
    model = _model()
    geometry = TorusGeometry(
        3,
        major_radius=100 / (2 * math.pi),
        minor_radius=1.0,
        representation=representation,
    ).double()
    model.chart.geometry = geometry
    # Sample all angles and varying cross-sections, including unsupported atoms.
    coordinates = torch.randn(251, 3, dtype=torch.float64)
    coordinates[:, 0] = torch.linspace(-150.0, 150.0, 251)
    center = geometry.encode_centers(geometry.lift_chart_coordinates(coordinates))
    p = torch.cat((torch.full((251, 1), 0.4), torch.ones(251, 1), center), dim=-1)
    owners = route_atoms(model.chart, model.kernel, p)
    touched = support_mask(model.chart, model.kernel, p)
    active = touched.any(0)
    assert bool(touched[owners[active], torch.arange(251)[active]].all())
    stations = torch.arange(model.chart.tile_count)[:, None]
    neighbors = (stations - owners[None, :]) % model.chart.tile_count
    assert not bool(
        (
            touched
            & (neighbors != 0)
            & (neighbors != 1)
            & (neighbors != model.chart.tile_count - 1)
        ).any()
    )
