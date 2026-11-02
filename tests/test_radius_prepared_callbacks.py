"""Prepared callback definitions must follow the current chart declarations."""

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import prepared
from torchcst._builtin_callbacks import original_attribute
from torchcst.charts.base import ChartSpec
from torchcst.charts.regular_grid import RegularGridChartSpec


def test_radius_callback_inventory_has_only_real_certified_attributes():
    callbacks = prepared._callbacks()
    assert prepared._definitions(callbacks) is not None
    assert not original_attribute(ChartSpec, "coordinate_shape")
    assert original_attribute(RegularGridChartSpec, "coordinate_shape")


def test_replaced_coordinate_shape_callback_cannot_be_prepared(monkeypatch):
    monkeypatch.setattr(
        RegularGridChartSpec, "coordinate_shape", property(lambda self: (42,))
    )
    assert prepared._definitions(prepared._callbacks()) is None
