import pytest
import torch
from torch import nn

from torchcst import Chart


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


def test_trainable_chart_registers_coordinates_as_a_parameter() -> None:
    chart = Chart.linspace(4, trainable=True)

    assert chart.trainable
    assert isinstance(chart.coordinates, nn.Parameter)
    assert tuple(chart.parameters()) == (chart.coordinates,)


def test_grid_has_one_point_per_cartesian_product_entry() -> None:
    chart = Chart.grid((2, 3), low=-1.0, high=1.0)

    assert chart.coordinates.shape == (6, 2)
    assert torch.equal(chart.coordinates[0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(chart.coordinates[-1], torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize("size", [0, -1])
def test_chart_sizes_must_be_positive(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Chart.linspace(size)


def test_explicit_coordinates_must_be_floating_point() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        Chart.points(torch.tensor([[0, 1]]))
