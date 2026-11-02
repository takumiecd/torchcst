"""Prepared callback definitions must follow the current chart declarations."""

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import prepared
from torchcst._builtin_callbacks import original_attribute
from torchcst.charts.base import ChartSpec
from torchcst.charts.regular_grid import RegularGridChartSpec


def test_radius_callback_inventory_has_only_real_certified_attributes():
    callbacks = prepared._callbacks()
    assert prepared._definitions(callbacks) is not None
    # Absence itself is deliberately certified for instance-owned attributes.
    assert original_attribute(ChartSpec, "coordinate_shape")
    # A function inventory still requires a real callback, not absence.
    assert prepared._definitions(((ChartSpec, ("coordinate_shape",)),)) is None
    assert original_attribute(RegularGridChartSpec, "coordinate_shape")


def test_replaced_coordinate_shape_callback_cannot_be_prepared(monkeypatch):
    monkeypatch.setattr(
        RegularGridChartSpec, "coordinate_shape", property(lambda self: (42,))
    )
    assert prepared._definitions(prepared._callbacks()) is None


def test_geometry_metadata_accessor_code_changes_cannot_be_prepared(monkeypatch):
    from torchcst import GeometryState
    from torchcst._backends.cuda.algorithms.polar_update import optimizer_program

    assert optimizer_program._definitions(optimizer_program._callbacks()) is not None

    def changed(self):
        return (123.0,)

    monkeypatch.setattr(GeometryState.periods.fget, "__code__", changed.__code__)
    assert prepared._definitions(prepared._callbacks()) is None
    assert optimizer_program._definitions(optimizer_program._callbacks()) is None
