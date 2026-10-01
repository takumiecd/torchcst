"""Snapshot public geometry/chart settings without loading evaluators.

Configuration only: scalar/coordinate reads can synchronize a device and
explicit tables can be large. Training execution keeps its live Tensor state.
"""

from .spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
    PointsPatternSpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


def geometry_declaration(geometry):
    from .geometry import EuclideanGeometry, SphereGeometry, TorusGeometry

    if type(geometry) is EuclideanGeometry:
        return EuclideanGeometrySpec(intrinsic_dim=geometry.intrinsic_dim)
    if type(geometry) is SphereGeometry:
        return SphereGeometrySpec(
            intrinsic_dim=geometry.intrinsic_dim,
            radius=float(geometry.radius.detach()),
            representation=geometry.representation,
            chart_margin=float(geometry.chart_margin.detach()),
        )
    if type(geometry) is TorusGeometry:
        return TorusGeometrySpec(
            intrinsic_dim=geometry.intrinsic_dim,
            major_radius=float(geometry.major_radius.detach()),
            minor_radius=float(geometry.minor_radius.detach()),
            circle_axis=geometry.circle_axis,
            max_arc_step=geometry.max_arc_step,
            representation=geometry.representation,
            chart_margin=float(geometry.chart_margin.detach()),
        )
    raise NotImplementedError("custom geometries must implement declaration()")


def _coordinates(tensor):
    return tuple(tuple(row) for row in tensor.detach().cpu().tolist())


def pattern_declaration(pattern):
    from .pattern import GridPattern, LinePattern, PointsPattern

    if type(pattern) in (GridPattern, LinePattern):
        cls = LinePatternSpec if type(pattern) is LinePattern else GridPatternSpec
        return cls(
            shape=pattern.shape,
            start=tuple(pattern.start.detach().cpu().tolist()),
            spacing=tuple(pattern.spacing.detach().cpu().tolist()),
        )
    if type(pattern) is PointsPattern:
        return PointsPatternSpec(coordinates=_coordinates(pattern.coordinates))
    raise NotImplementedError("custom site patterns must implement declaration()")


def chart_declaration(chart):
    from .chart import ExplicitChart
    from .lazy_chart import ProductChart, StripChart

    geometry = chart.geometry.declaration()
    if type(chart) is ExplicitChart:
        spacing = (
            None
            if chart.spacing is None
            else tuple(chart.spacing.detach().cpu().tolist())
        )
        return ChartSpec(
            kind="explicit",
            geometry=geometry,
            shape=chart.shape,
            axes=(PointsPatternSpec(coordinates=_coordinates(chart.coordinates)),),
            trainable=chart.trainable,
            spacing=spacing,
        )
    if type(chart) in (ProductChart, StripChart):
        values = {
            "geometry": geometry,
            "shape": chart.shape,
            "axes": tuple(a.declaration() for a in chart.axes),
        }
        if type(chart) is ProductChart:
            return ChartSpec(kind="product", **values)
        return ChartSpec(
            kind="strip",
            **values,
            tile_shape=chart.tile_shape,
            axis=chart.axis,
            tile_pitch=float(chart.tile_pitch.detach()),
        )
    raise NotImplementedError("custom charts must implement declaration()")
