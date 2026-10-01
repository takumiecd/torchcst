"""Functional evaluation of small site patterns."""

from torchcst.geometry.spec import GridPatternSpec, LinePatternSpec, PointsPatternSpec
from torchcst.geometry.state import PatternState

from . import grid, points


def _module(state):
    if type(state) is not PatternState or state.spec.revision != 1:
        raise TypeError("pattern must be a PatternState")
    if type(state.spec) in (GridPatternSpec, LinePatternSpec):
        return grid
    if type(state.spec) is PointsPatternSpec:
        return points
    raise ValueError("unsupported pattern declaration")


def positions(state, indices):
    return _module(state).positions(state, indices)


def bounds(state):
    return _module(state).bounds(state)
