"""Fixed geometry/grid metadata survives placement, reload and cache changes."""

import copy
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_radius_polar_cuda_update import radius_model

from torchcst import GeometryState, compile_chart, geometry_presets
from torchcst import chart_presets as charts
from torchcst._backends.cuda.algorithms.polar_update.algorithm import _period_tensors
from torchcst._backends.torch.charts import execution
from torchcst.operators.metadata import fixed_state_signature


@pytest.mark.parametrize(
    "geometry",
    [
        geometry_presets.euclidean(2),
        geometry_presets.sphere(2, radius=0.37),
        geometry_presets.flat_torus((0.37, 1.23)),
    ],
)
def test_geometry_has_no_fixed_tensors_and_keeps_full_precision(geometry):
    state = GeometryState(geometry)
    assert not tuple(state.buffers()) and not tuple(state.parameters())
    assert not any(isinstance(v, torch.Tensor) for v in vars(state).values())
    before = state.declaration()
    state.half().float().double()
    assert state.declaration() == before == geometry
    stream = io.BytesIO()
    torch.save(state.state_dict(), stream)
    stream.seek(0)
    other = GeometryState(geometry)
    other.load_state_dict(torch.load(stream, weights_only=True))
    assert other.declaration() == geometry


def test_regular_grid_is_tensor_free_and_runtime_centers_own_placement():
    spec = charts.regular_grid((3, 4), spacing=(0.37, 0.23), origin=(-1.2, 0.31))
    state = compile_chart(spec, dtype=torch.float16)
    assert not tuple(state.buffers()) and not tuple(state.parameters())
    state.float().double().half()
    assert state.declaration() == spec
    center = torch.tensor([[0.13, -0.21]], dtype=torch.float64, requires_grad=True)
    actual = execution.squared_distance(state, center)
    sites = torch.tensor(
        [
            (spec.origin[0] + i * spec.spacing[0], spec.origin[1] + j * spec.spacing[1])
            for i in range(3)
            for j in range(4)
        ],
        dtype=center.dtype,
    )
    expected = (sites[:, None] - center).square().sum(-1)
    assert actual.dtype == center.dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), center, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), center)[0],
    )
    clone = copy.deepcopy(state)
    clone.load_state_dict(state.state_dict())
    assert clone.declaration() == spec
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        compile_chart(replace(spec, spacing=(0.1, 0.2))).load_state_dict(
            state.state_dict()
        )


def test_period_constants_are_transient_follow_atoms_and_rebuild():
    model = radius_model()
    state = SimpleNamespace(
        binding=SimpleNamespace(cst_charts=model.cst_charts), buffers={}
    )
    p = model.atoms.p
    first = _period_tensors(state, p)
    assert _period_tensors(state, p) is first
    assert all(v.dtype == p.dtype and v.device == p.device for v in first)
    assert not tuple(model.input_chart.geometry.buffers())
    model.input_chart.geometry.double()  # Geometry placement hints never own execution dtype.
    assert _period_tensors(state, p) is first
    promoted = _period_tensors(state, p.double())
    assert all(v.dtype == torch.float64 for v in promoted)
    old = tuple(v.clone() for v in promoted)
    for chart in {id(c): c for c in model.cst_charts()}.values():
        geom = replace(
            chart.geometry.spec, periods=tuple(v * 2 for v in chart.geometry.periods)
        )
        chart.geometry.spec = geom
        chart.spec = replace(
            chart.spec, geometry=geom, spacing=tuple(v * 2 for v in chart.spacing)
        )
    new = _period_tensors(state, p.double())
    for a, b in zip(new, old):
        torch.testing.assert_close(a, 2 * b)
    new[0].zero_()  # A changed internal constant cannot stay certified.
    repaired = _period_tensors(state, p.double())
    torch.testing.assert_close(repaired[0], old[0] * 2)
    assert not any("periods" in key for key in model.state_dict())


def test_metadata_signature_detects_even_forced_edits_to_a_frozen_spec():
    model = radius_model()
    roots = (*model.cst_charts(), model.kernel)
    before = fixed_state_signature(roots)
    chart = model.input_chart
    object.__setattr__(chart.spec, "origin", (0.125,))
    after = fixed_state_signature(roots)
    assert after != before
