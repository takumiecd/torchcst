"""Mode4 local moments coexist with mode3 shared and mode5 global host graphs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.training.cell_mass_mnist import make_radius_cst
from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
    GLOBAL_ACTIVE_GRAD4,
    GLOBAL_ACTIVE_GRAD4_NATIVE,
    LOCAL_TILE_MOMENTS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_local_tile_comparison_cases_are_opt_in_and_keep_global_native5():
    for name, rows in [
        ("radius-cell-mass-128-local-tile-initial", 4),
        ("radius-cell-mass-128-local-tile-batch32-initial", 32),
    ]:
        run = load_run(
            ROOT / f"benchmarks/cuda/linear/cases/{name}.json",
            ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
        )
        assert run.baseline == "radius-native-batched-cached-parallel"
        assert run.entry("radius-native-local-tile-moments").plan == LOCAL_TILE_MOMENTS
        assert run.case.rows == rows and run.case.atoms == 819 and run.dense
        assert decode_snapshot(run.snapshot()) == run
    registry = make_algorithm_registry()
    assert (
        registry.loads_plan(registry.dumps_plan(GLOBAL_ACTIVE_GRAD4_NATIVE))
        == GLOBAL_ACTIVE_GRAD4_NATIVE
    )
    assert (
        GLOBAL_ACTIVE_GRAD4_NATIVE.recipe.schedule
        == "global_batched_cached_active_moments_grad4_native"
    )


@pytest.mark.parametrize("mode", [3, 4, 5])
def test_checked_wrapper_selects_distinct_host_mode_with_live_axes(monkeypatch, mode):
    selected = {
        3: replace(
            BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
            recipe=replace(
                BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
                native_autograd=True,
                native_entry=True,
            ),
        ),
        4: LOCAL_TILE_MOMENTS,
        5: GLOBAL_ACTIVE_GRAD4_NATIVE,
    }[mode]
    width = 1024 if mode == 5 else 128
    layer = make_radius_cst(width, 11, execution_plan=selected)
    plan = hierarchy_from_operator(layer.declaration())
    x = torch.zeros(5, width)
    calls = []

    def invoke(*args):
        calls.append(args)
        return x

    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear=invoke, checked_autograd_linear_local_tile=invoke
        ),
    )
    assert ex.checked_radius_linear(plan, x, layer.atoms.p, recipe=selected.recipe) is x
    assert len(calls) == 1
    args = calls[0]
    assert args[0] is x and args[1] is layer.atoms.p
    assert args[2:4] == (plan.input.native_config(), plan.output.native_config())
    assert args[-1] == (32 if mode == 4 else mode)
    assert len(args) == (8 if mode == 4 else 9)


@GPU
@pytest.mark.parametrize("mode", [4, 5])
def test_atom_only_keys_are_saved_for_mode4_and_omitted_for_global5(mode):
    selected = LOCAL_TILE_MOMENTS if mode == 4 else GLOBAL_ACTIVE_GRAD4_NATIVE
    width = 128 if mode == 4 else 1024
    layer = make_radius_cst(width, 11, dtype=torch.float64, execution_plan=selected)
    plan = hierarchy_from_operator(layer.declaration())
    x = torch.randn(5, width, device="cuda", dtype=torch.float64)
    p = layer.atoms.p.detach().cuda().requires_grad_()
    saved = []

    def pack(value):
        saved.append((value.dtype, tuple(value.shape)))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        y = ex.radius_linear(plan, x, p, recipe=selected.recipe)
    assert len(saved) == (5 if mode == 4 else 4)
    assert ((torch.int64, (11,)) in saved) == (mode == 4)
    actual = torch.autograd.grad(y, p, torch.ones_like(y))[0]
    reference = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4.recipe)
    expected = torch.autograd.grad(reference, p, torch.ones_like(reference))[0]
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("entry", ["checked", "radius", "algorithm"])
@pytest.mark.parametrize(
    "schedule,mode",
    [
        ("block_local", 0),
        ("block_local_parallel", 1),
        ("block_local_cached_parallel", 2),
        ("block_local_batched_cached_parallel", 3),
        ("block_local_tile_moments", 4),
        ("global_batched_cached_active_moments_grad4_native", 5),
        ("global_owner_response_coefficients", 6),
        ("global_owner_response_derivatives", 7),
        ("global_owner_compact_partials", 9),
    ],
)
def test_all_real_entry_boundaries_preserve_local4_and_primary_legacy_modes(
    monkeypatch, entry, schedule, mode
):
    import torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm as algorithm_module
    from torchcst import LinearInputs
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        RadiusCellMassAlgorithm,
    )
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        GLOBAL_OWNER_COMPACT_PARTIALS,
    )

    layer = make_radius_cst(128, 11)
    layout = hierarchy_from_operator(layer.declaration())
    x, p = torch.zeros(4, 128), layer.atoms.p
    recipe = replace(GLOBAL_OWNER_COMPACT_PARTIALS.recipe, schedule=schedule)
    calls = []

    def old_native(*args):
        calls.append((args[-1], args))
        return x

    def local_native(*args):
        calls.append((4, args))
        return x

    def compact_native(*args):
        calls.append((9, args))
        return x

    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear=old_native,
            checked_autograd_linear_local_tile=local_native,
            checked_autograd_linear_compact_partials=compact_native,
        ),
    )
    if entry == "algorithm":
        state = SimpleNamespace(layout=layout, recipe=recipe)
        inputs = LinearInputs(x)

        def execution(actual_state, actual_inputs):
            assert actual_state is state and actual_inputs is inputs
            return x, p, None, None

        monkeypatch.setattr(algorithm_module, "linear_execution", execution)
        actual = RadiusCellMassAlgorithm().execute(state, inputs)
    else:
        method = ex.checked_radius_linear if entry == "checked" else ex.radius_linear
        actual = method(layout, x, p, recipe=recipe)
    assert actual is x and len(calls) == 1
    actual_mode, args = calls[0]
    assert actual_mode == mode and args[0] is x and args[1] is p
    assert args[2:4] == (layout.input.native_config(), layout.output.native_config())
    assert len(args) == (8 if mode == 4 else 4 if mode == 9 else 9)
