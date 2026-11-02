"""Implicit regular grid layout and geometry contracts."""

import copy
import itertools

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    GaussianSpec,
    RegularGridChartSpec,
    compile_chart,
    kernels,
)
from torchcst import (
    chart_presets as charts,
)
from torchcst import (
    geometry_presets as geometries,
)
from torchcst._backends.torch.charts import execution as chart_ops


def kernel(sigma=0.8):
    return kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=2.0,
            input_bounds=BandwidthBounds(
                minimum=sigma, birth=sigma, maximum=sigma, upper_floor=sigma
            ),
            w_c=0.5,
        ),
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_broadcast_indexing_ncs_and_bounded_storage(periodic):
    spec = charts.regular_grid(
        (2, 3, 4),
        spacing=(0.5, 1.0, 2.0),
        origin=-0.5,
        geometry="flat_torus" if periodic else "euclidean",
    )
    assert isinstance(spec, RegularGridChartSpec)
    assert spec.shape == (2, 3, 4)
    assert spec.coordinate_shape == (2, 3, 4)
    assert spec.features == 24
    assert spec.num_sites == 24
    assert spec.ncs == 24
    assert spec.origin == (-0.5,) * 3
    state = compile_chart(spec, dtype=torch.float64)
    assert state.features == 24
    assert state.num_sites == 24
    assert state.ncs == 24
    sites = torch.tensor(
        list(itertools.product((-0.5, 0.0), (-0.5, 0.5, 1.5), (-0.5, 1.5, 3.5, 5.5)))
    )
    ids = torch.tensor([0, 1, 4, 11, 12, 23, 4])
    torch.testing.assert_close(chart_ops.positions(state, ids), sites[ids].double())
    assert sum(v.numel() for v in state.buffers()) == 0
    large = compile_chart(charts.regular_grid((10000, 20000, 30000), spacing=0.5))
    assert sum(v.numel() for v in large.buffers()) == 0
    assert chart_ops.positions(state, torch.empty(0, dtype=torch.long)).shape == (0, 3)
    if periodic:
        assert spec.geometry.periods == (1.0, 3.0, 8.0)
    assert charts.regular_grid(256).shape == (256,)
    assert charts.regular_grid(256).ncs == 256
    assert charts.regular_grid((16, 8)).shape == (16, 8)
    assert charts.regular_grid((16, 8)).ncs == 128
    assert charts.regular_grid((2, 3, 4)).shape == (2, 3, 4)
    assert charts.regular_grid((2,)).shape == (2,)
    for invalid in (torch.tensor([-1]), torch.tensor([24])):
        with pytest.raises(IndexError):
            chart_ops.positions(state, invalid)
    with pytest.raises(ValueError):
        chart_ops.positions(state, torch.tensor([0.0]))


@pytest.mark.parametrize(
    "settings",
    [
        {"grid_shape": ()},
        {"grid_shape": ((3,), (4,))},
        {"grid_shape": (3, ())},
        {"grid_shape": (True, 4)},
        {"grid_shape": (0, 4)},
        {"grid_shape": 0},
        {"grid_shape": True},
        {"spacing": 0},
        {"spacing": -1},
        {"spacing": True},
        {"spacing": float("inf")},
        {"spacing": (1,)},
        {"spacing": (1, float("nan"))},
        {"spacing": (1, True)},
        {"origin": (0,)},
        {"origin": float("inf")},
        {"origin": True},
        {"geometry": geometries.euclidean(3)},
        {"geometry": geometries.sphere(2)},
        {"geometry": "unknown"},
    ],
)
def test_invalid_declarations(settings):
    values = {"grid_shape": (3, 4)}
    values.update(settings)
    with pytest.raises((ValueError, TypeError)):
        charts.regular_grid(**values)


def test_scalar_spacing_is_equivalent_to_per_coordinate_values():
    for geometry in (None, "flat_torus"):
        assert charts.regular_grid(
            (16, 16, 128), spacing=0.1, geometry=geometry
        ) == charts.regular_grid(
            (16, 16, 128), spacing=(0.1, 0.1, 0.1), geometry=geometry
        )
        assert charts.regular_grid(64, spacing=0.1, geometry=geometry) == (
            charts.regular_grid((64,), spacing=(0.1,), geometry=geometry)
        )


def test_integer_origin_uses_floating_state_and_large_index_uniform_bounds():
    state = compile_chart(charts.regular_grid((3, 4), origin=0))
    assert state.dtype.is_floating_point
    assert state.origin == (0.0, 0.0)
    assert state.spacing == (1.0, 1.0)
    large = compile_chart(
        charts.regular_grid((100000, 3), spacing=0.0002), dtype=torch.float16
    )
    indices = torch.tensor([99999 * 3, 99999 * 3 + 2])
    sites = chart_ops.positions(large, indices)
    assert torch.isfinite(sites).all()
    centers = chart_ops.initialize_centers(large, 8, mode="uniform")
    assert torch.isfinite(centers).all()
    wide = compile_chart(
        charts.regular_grid((5, 1), spacing=30000, origin=(-60000, 0)),
        dtype=torch.float16,
    )
    assert torch.isfinite(chart_ops.initialize_centers(wide, 8, mode="uniform")).all()


@pytest.mark.parametrize("periodic", [False, True])
def test_checkpoint_layout_mismatch_and_live_buffer_validation(periodic):
    spec = charts.regular_grid(
        (2, 3, 4), spacing=0.5, geometry="flat_torus" if periodic else None
    )
    state = compile_chart(spec, dtype=torch.float64)
    saved = copy.deepcopy(state.state_dict())
    restored = compile_chart(spec, dtype=torch.float32).double()
    restored.load_state_dict(saved)
    assert restored.declaration() == state.declaration()
    with pytest.raises(RuntimeError):
        compile_chart(
            charts.regular_grid((3, 2, 4), geometry="flat_torus" if periodic else None)
        ).load_state_dict(saved)
    for name, bad in (("spacing", 0.0), ("origin", float("nan"))):
        broken = copy.deepcopy(saved)
        broken["_extra_state"][name] = (bad, *broken["_extra_state"][name][1:])
        with pytest.raises(RuntimeError):
            restored.load_state_dict(broken)
    tiny = compile_chart(
        charts.regular_grid((3, 4), spacing=1e-20), dtype=torch.float16
    )
    assert tiny.spacing == (1e-20, 1e-20)  # Metadata is not rounded by placement.
    huge = compile_chart(charts.regular_grid((3, 4), origin=1e20), dtype=torch.float16)
    with pytest.raises(ValueError, match="not finite"):
        chart_ops.positions(huge, torch.tensor([0]))


def test_public_optimizer_wraps_periodic_centers_and_preserves_euclidean_centers():
    for periodic in (False, True):
        in_spec = charts.regular_grid(4, geometry="flat_torus" if periodic else None)
        out_spec = charts.regular_grid(3, geometry="flat_torus" if periodic else None)
        p = torch.tensor([[0.2, 1.2, -0.3, 4.3]], dtype=torch.float64)
        model = CSTLinear(
            in_spec,
            out_spec,
            atoms=p,
            kernel=kernel(),
            backend="factored",
            dtype=p.dtype,
        )
        optimizer = CSTOptimizer(
            torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0), model=model
        )
        model.atoms.p.grad = torch.zeros_like(model.atoms.p)
        optimizer.step()
        centers = model.atoms.p[:, 2:]
        if periodic:
            periods = torch.tensor(
                [c.geometry.periods[0] for c in model.cst_charts()], dtype=p.dtype
            )
            torch.testing.assert_close(centers, torch.remainder(p[:, 2:], periods))
        else:
            torch.testing.assert_close(centers, p[:, 2:])
