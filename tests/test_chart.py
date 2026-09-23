import pytest
import torch
from torch import nn

from torchcst import Chart, EuclideanGeometry, SphereGeometry


def test_points_owns_a_fixed_shape_copy() -> None:
    coordinates = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    chart = Chart.points(coordinates)

    coordinates.add_(10)

    assert chart.features == 2
    assert chart.dim == 2
    assert chart.coordinates.shape == (2, 2)
    assert torch.equal(chart.coordinates, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert dict(chart.named_parameters()) == {}
    assert "coordinates" in dict(chart.named_buffers())
    assert chart.spacing is None
    assert isinstance(chart.geometry, EuclideanGeometry)
    assert chart.intrinsic_dim == chart.embedding_dim == 2


def test_trainable_chart_registers_coordinates_as_a_parameter() -> None:
    chart = Chart.linspace(4, spacing=1.0, trainable=True)

    assert chart.trainable
    assert isinstance(chart.coordinates, nn.Parameter)
    assert tuple(chart.parameters()) == (chart.coordinates,)


def test_linspace_spacing_is_centered() -> None:
    chart = Chart.linspace(64, spacing=0.10)

    assert chart.dim == 1
    assert chart.features == 64
    assert torch.allclose(chart.spacing, torch.tensor([0.10]))
    assert torch.allclose(chart.coordinates[0], torch.tensor([-3.15]))
    assert torch.allclose(chart.coordinates[-1], torch.tensor([3.15]))
    assert torch.allclose(
        chart.coordinates[1] - chart.coordinates[0], torch.tensor([0.10])
    )


def test_linspace_low_high_stores_derived_spacing() -> None:
    chart = Chart.linspace(5, low=-1.0, high=1.0)

    assert torch.allclose(chart.coordinates[0], torch.tensor([-1.0]))
    assert torch.allclose(chart.coordinates[-1], torch.tensor([1.0]))
    assert torch.allclose(chart.spacing, torch.tensor([0.5]))


def test_grid_spacing_is_isotropic_and_centered() -> None:
    chart = Chart.grid((8, 8), spacing=0.10)

    assert chart.coordinates.shape == (64, 2)
    assert torch.allclose(chart.spacing, torch.tensor([0.10, 0.10]))
    assert torch.allclose(chart.coordinates[0], torch.tensor([-0.35, -0.35]))
    assert torch.allclose(chart.coordinates[-1], torch.tensor([0.35, 0.35]))


def test_grid_low_high_has_one_point_per_cartesian_product_entry() -> None:
    chart = Chart.grid((2, 3), low=-1.0, high=1.0)

    assert chart.coordinates.shape == (6, 2)
    assert torch.equal(chart.coordinates[0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(chart.coordinates[-1], torch.tensor([1.0, 1.0]))


def test_grid_accepts_per_axis_spacing() -> None:
    chart = Chart.grid((2, 3), spacing=(1.0, 0.5))

    assert torch.allclose(chart.spacing, torch.tensor([1.0, 0.5]))
    xs = chart.coordinates[:, 0].unique(sorted=True)
    ys = chart.coordinates[:, 1].unique(sorted=True)
    assert torch.allclose(xs[1] - xs[0], torch.tensor(1.0))
    assert torch.allclose(ys[1] - ys[0], torch.tensor(0.5))


@pytest.mark.parametrize("size", [0, -1])
def test_chart_sizes_must_be_positive(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Chart.linspace(size, spacing=1.0)


def test_explicit_coordinates_must_be_floating_point() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        Chart.points(torch.tensor([[0, 1]]))


def test_linspace_requires_spacing_or_endpoints() -> None:
    with pytest.raises(ValueError, match="spacing or low and high"):
        Chart.linspace(4)


def test_linspace_rejects_mixed_layout() -> None:
    with pytest.raises(ValueError, match="not both"):
        Chart.linspace(4, spacing=0.1, low=-1.0, high=1.0)


def test_linspace_rejects_center_with_endpoints() -> None:
    with pytest.raises(ValueError, match="center is only used with spacing"):
        Chart.linspace(4, low=-1.0, high=1.0, center=0.0)


def test_spacing_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Chart.linspace(4, spacing=0.0)


def test_chart_rejects_coordinates_outside_its_geometry() -> None:
    with pytest.raises(ValueError, match="must lie on"):
        Chart.points(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            geometry=SphereGeometry(2),
        )
