"""Explicit row reuse candidate: declarations and unchanged native arithmetic."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_ffn_rectangular_2048 import context

from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.row8 import (
    ROW8_NATIVE,
    RadiusRow8Algorithm,
)
from torchcst._backends.schema import DeviceInfo


@pytest.mark.parametrize("ni,no", [(256, 1024), (1024, 256)])
@pytest.mark.parametrize("rows", [1, 4, 7, 8, 9, 255, 256])
def test_explicit_support_and_roundtrip(ni, no, rows):
    ctx = replace(
        context(ni, no, count=4096), input_shape=(rows, ni), input_strides=(ni, 1)
    )
    algorithm = RadiusRow8Algorithm()
    assert algorithm.supports(ctx, ROW8_NATIVE.recipe).supported
    registry = make_algorithm_registry()
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(ROW8_NATIVE)) == ROW8_NATIVE


@pytest.mark.parametrize(
    "change",
    [
        {"input_shape": (257, 256)},
        {"input_shape": (0, 256)},
        {"atom_count": 4095},
        {"dtype": torch.float64},
        {"device": DeviceInfo("cpu", None, "metadata-only")},
    ],
)
def test_reject_unqualified_conditions(change):
    ctx = context(256, 1024, count=4096)
    assert (
        not RadiusRow8Algorithm()
        .supports(replace(ctx, **change), ROW8_NATIVE.recipe)
        .supported
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ni": 512, "no": 2048, "count": 8192},
        {"modes": 3},
        {"spacing": 0.5},
        {"maximum": 51},
    ],
)
def test_reject_unqualified_geometry(kwargs):
    values = {"ni": 256, "no": 1024, "count": 4096}
    values.update(kwargs)
    assert (
        not RadiusRow8Algorithm()
        .supports(context(**values), ROW8_NATIVE.recipe)
        .supported
    )


def test_candidate_is_not_in_production_catalog():
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(ROW8_NATIVE)
    with pytest.raises(ValueError):
        RadiusRow8Algorithm().validate_recipe(
            replace(ROW8_NATIVE.recipe, native_entry=False)
        )


def test_explicit_binary_selection(monkeypatch):
    ctx = context(256, 1024, count=4096)
    plan = hierarchy_from_operator(ctx.operator)
    x, p = torch.zeros(9, 256), torch.zeros(4096, 4)
    calls = []

    def extension(*args):
        calls.append(args)
        return SimpleNamespace(checked_autograd_linear=lambda *a: x)

    monkeypatch.setattr(ex, "_extension", extension)
    assert (
        ex.checked_radius_linear(plan, x, p, recipe=ROW8_NATIVE.recipe, _row_batch=8)
        is x
    )
    assert calls == [(8,)]
    calls.clear()
    assert (
        ex.checked_radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
        is x
    )
    assert calls == [()]


@pytest.mark.parametrize("value", [True, 0, 16, "8"])
def test_invalid_build_width_rejected_without_compilation(value):
    with pytest.raises(ValueError):
        ex._extension(value)
