"""Explicit row reuse candidate: declarations and unchanged native arithmetic."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_ffn_rectangular_2048 import context

from torchcst import (
    ChebyshevSpec,
    CSTLinear,
    FixedSelector,
    RadiusBounds,
    chart_presets,
    kernels,
    make_algorithm_registry,
)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("ni,no", [(256, 1024), (1024, 256)])
@pytest.mark.parametrize("rows", [7, 8, 9, 255, 256])
def test_cuda_tail_rows_periodic_geometry_retained_vjp_and_versions(ni, no, rows):
    torch.manual_seed(1701)
    m = CSTLinear(
        chart_presets.regular_grid(no, spacing=1, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1, geometry="flat_torus"),
        atoms=4096,
        dtype=torch.float32,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.polar(
                input_bounds=RadiusBounds(
                    minimum=0.5, birth=4, maximum=50, upper_floor=1
                )
            ),
        ),
    )
    with torch.no_grad():
        ratios = torch.linspace(-0.999, 0.999, 4096)
        ratios[:4] = torch.tensor([0, 1e-7, -1e-7, 0.3])
        rho = (1 + 3 * (torch.arange(4096) % 4) / 3).sqrt()
        m.atoms.p[:, 0] = ratios * rho
        m.atoms.p[:, 1] = (1 - ratios.square()).sqrt() * rho
        m.atoms.p[:, 2] = torch.linspace(-0.5, no + 0.5, 4096)
        m.atoms.p[:, 3] = torch.linspace(ni + 0.5, -0.5, 4096)
    registry = make_algorithm_registry()
    registry.register(RadiusRow8Algorithm())
    a, b = copy.deepcopy(m).cuda(), copy.deepcopy(m).cuda()
    a.selector = FixedSelector(GLOBAL_ACTIVE_GRAD4_NATIVE, registry=registry)
    b.selector = FixedSelector(ROW8_NATIVE, registry=registry)
    x = torch.randn(rows, ni, device="cuda")
    dy = torch.randn(rows, no, device="cuda")
    outputs = []
    for layer in [a, b]:
        xx = x.clone().requires_grad_()
        y = layer(xx)
        gx, gp = torch.autograd.grad(y, (xx, layer.atoms.p), dy, retain_graph=True)
        again = torch.autograd.grad(y, (xx, layer.atoms.p), dy, retain_graph=True)
        for left, right in zip((gx, gp), again):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        outputs.append((y, gx, gp))
        if layer is b:
            with torch.no_grad():
                layer.atoms.p.add_(0.0001)
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                torch.autograd.grad(y, (xx, layer.atoms.p), dy)
    for left, right in zip(*outputs):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)
