"""Global C++ autograd changes host dependencies, never the arithmetic kernels."""

import copy
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_batched_cached import bounded_atoms
from test_radius_cell_mass_cuda import GPU, axis_plan
from test_radius_cell_mass_hybrid import check_vjp
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.cuda.linear.radius_cell_mass import configure, fixture, optimizer
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
    GLOBAL_ACTIVE_GRAD4,
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)

ROOT = Path(__file__).resolve().parents[1]


def run():
    return load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-global-native-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_explicit_global_native_roundtrip_caps_and_old_flags():
    loaded = run()
    assert (
        loaded.entry("radius-global-native-active-grad4").plan
        == GLOBAL_ACTIVE_GRAD4_NATIVE
    )
    assert loaded.baseline == "radius-global-active-grad4"
    assert decode_snapshot(loaded.snapshot()) == loaded
    candidate = GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
    assert (
        replace(
            candidate,
            schedule=GLOBAL_ACTIVE_GRAD4.recipe.schedule,
            native_autograd=False,
            native_entry=False,
        )
        == GLOBAL_ACTIVE_GRAD4.recipe
    )
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(metadata_context(), candidate).supported
    assert algorithm.supports(metadata_context(count=65536), candidate).supported
    assert not algorithm.supports(metadata_context(count=65537), candidate).supported
    assert not algorithm.supports(
        replace(metadata_context(), input_shape=(65536, 1024)), candidate
    ).supported
    # General mode never loosens the old all-atom shared projection limit.
    assert not algorithm.supports(
        metadata_context(), BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe
    ).supported
    for changes in (
        {"native_autograd": True},
        {"native_autograd": True, "native_entry": True},
    ):
        with pytest.raises(ValueError, match="native_autograd"):
            algorithm.validate_recipe(replace(GLOBAL_ACTIVE_GRAD4.recipe, **changes))
    with pytest.raises(ValueError, match="native_entry"):
        algorithm.validate_recipe(replace(candidate, native_autograd=False))


def test_checked_cpu_wrapper_passes_live_arguments_and_distinct_mode(monkeypatch):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    x, p = torch.zeros(5, 1024), torch.zeros(41, 4)
    calls = []

    def native(*args):
        calls.append(args)
        return x

    monkeypatch.setattr(
        ex, "_extension", lambda: SimpleNamespace(checked_autograd_linear=native)
    )
    assert (
        ex.checked_radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
        is x
    )
    assert calls == [
        (x, p, axis.native_config(), axis.native_config(), 2, 2, True, 32, 5)
    ]


def test_checked_tensor_subclass_retains_python_fallback(monkeypatch):
    class Custom(torch.Tensor):
        pass

    x = torch.zeros(1, 1024).as_subclass(Custom)
    p = torch.zeros(3, 4)
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return x

    monkeypatch.setattr(ex, "radius_linear", fallback)
    assert (
        ex.checked_radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
        is x
    )
    assert calls[0][0] == (plan, x, p)
    assert not calls[0][1]["recipe"].native_entry
    assert calls[0][1]["recipe"].native_autograd


def test_actual_host_autograd_compiles_against_installed_torch_headers():
    from torch.utils.cpp_extension import include_paths

    compiler = shlex.split(os.getenv("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("host C++ compiler unavailable")
    source = Path(ex.__file__).with_name("host_autograd.cpp")
    command = [
        *compiler,
        "-std=c++17",
        "-fsyntax-only",
        "-DTORCH_EXTENSION_NAME=radius_host_syntax",
        str(source),
    ]
    for path in [*include_paths(), sysconfig.get_paths()["include"]]:
        command.extend(["-I", path])
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,start,method",
    [(128, 2, "merge"), (1024, 2, "merge"), (1024, 5, "direct"), (1024, 11, "merge")],
)
def test_global_native_independent_physical_all_vjps(dtype, sites, start, method):
    axis = axis_plan(sites, spacing=0.37, origin=-1.2)
    plan = HierarchyPlan(input=axis, output=axis)
    gen = torch.Generator().manual_seed(1481)
    check_vjp(
        plan,
        replace(
            GLOBAL_ACTIVE_GRAD4_NATIVE.recipe,
            input_direct_levels=start,
            output_direct_levels=start,
            moment_method=method,
        ),
        torch.randn(5, sites, dtype=dtype, generator=gen),
        bounded_atoms(plan, dtype),
        torch.randn(5, sites, dtype=dtype, generator=gen),
    )


@GPU
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
def test_global_native_identical_bits_hooks_retained_and_atom_only(
    checked, offload, needs_x, needs_p
):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(5, 1024, device="cuda", dtype=torch.float64, requires_grad=needs_x)
    p = bounded_atoms(plan, torch.float64).cuda().requires_grad_(needs_p)
    dy = torch.randn_like(x)
    selected = tuple(v for v in (x, p) if v.requires_grad)
    seen = []

    def pack(value):
        seen.append(tuple(value.shape))
        return (value.detach().cpu().clone(), value.device) if offload else value

    def unpack(saved):
        return saved[0].to(saved[1]) if offload else saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = ex.radius_linear(
            plan,
            x,
            p,
            recipe=replace(GLOBAL_ACTIVE_GRAD4_NATIVE.recipe, native_entry=checked),
        )
    assert "RadiusLinear" in y.grad_fn.name()
    assert len(seen) == (5 if needs_x else 4)
    actual = torch.autograd.grad(y, selected, dy, retain_graph=True)
    ref = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4.recipe)
    expected = torch.autograd.grad(ref, selected, dy)
    for a, b in zip((y, *actual), (ref, *expected)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    again = torch.autograd.grad(y, selected, dy)
    for a, b in zip(actual, again):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@GPU
@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference"])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(False, False), (True, False), (False, True), (True, True)]
)
def test_global_native_forward_without_edges(mode, needs_x, needs_p):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(4, 1024, device="cuda", requires_grad=needs_x)
    p = bounded_atoms(plan, torch.float32).cuda().requires_grad_(needs_p)
    context = {
        "grad": torch.enable_grad,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }[mode]
    with context():
        actual = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
        expected = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4.recipe)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert (
        actual.requires_grad
        == expected.requires_grad
        == (mode == "grad" and (needs_x or needs_p))
    )
    assert actual.is_inference() == expected.is_inference() == (mode == "inference")


@GPU
@pytest.mark.parametrize(
    "wrong", ["atoms", "sites", "batch", "level", "tile", "mode4", "shared_mode"]
)
def test_global_native_metadata_rejected_before_any_device_kernel(wrong):
    axis = axis_plan(1024)
    config = axis.native_config()
    p = torch.zeros(65537 if wrong == "atoms" else 8, 4, device="cuda")
    x = torch.zeros(65536 if wrong == "batch" else 4, 1024, device="cuda")
    start, tile, mode = 2, 32, 5
    if wrong == "sites":
        config = [2048, *config[1:]]
    elif wrong == "level":
        start = -1
    elif wrong == "tile":
        tile = 0
    elif wrong == "mode4":
        mode = 4
    elif wrong == "shared_mode":
        mode = 3
    native = ex._extension()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with pytest.raises(RuntimeError):
            native.checked_autograd_linear(
                x, p, config, config, start, start, True, tile, mode
            )
    assert not any(str(event.device_type).endswith(".CUDA") for event in prof.events())


@GPU
def test_global_native_version_check_and_higher_derivative_rejection():
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(4, 1024, device="cuda", dtype=torch.float64, requires_grad=True)
    p = bounded_atoms(plan, torch.float64).cuda().requires_grad_()
    y = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
    with torch.no_grad():
        p.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()
    y = ex.radius_linear(plan, x, p, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)


@GPU
@pytest.mark.parametrize("batch", [4, 5])
def test_global_native_public_polar_three_step_state_and_read_leases(batch):
    loaded = run()
    case = replace(loaded.case, rows=batch, atoms=41)
    cpu, host_x, target, _ = fixture(case)
    records = []
    for selected in [GLOBAL_ACTIVE_GRAD4, GLOBAL_ACTIVE_GRAD4_NATIVE]:
        layer = configure(copy.deepcopy(cpu).cuda(), selected)
        opt = optimizer(layer, case)
        x = host_x.cuda().requires_grad_()
        dy = torch.ones(batch, 1024, device="cuda")
        calls = []
        handle = layer.register_forward_hook(lambda *args: calls.append("forward"))
        hook = x.register_hook(lambda g: g * 0.5)
        y = layer(x)
        torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
        with pytest.raises(RuntimeError, match="outstanding|retained"):
            layer.atom_state._require_boundary()
        torch.autograd.grad(y, (x, layer.atoms.p), dy)
        layer.atom_state._require_boundary()
        assert calls == ["forward"]
        handle.remove()
        hook.remove()
        values = {}
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            x.grad = None
            y = layer(x)
            (y - target.cuda()).square().mean().backward()
            for name, v in [("Y", y), ("dX", x.grad), ("dP", layer.atoms.p.grad)]:
                values[step, name] = v.detach().cpu().clone()
            opt.step()
            values[step, "P"] = layer.atoms.p.detach().cpu().clone()
            for key, value in opt.state[layer.atoms.p].items():
                if torch.is_tensor(value):
                    values[step, key] = value.detach().cpu().clone()
        records.append(values)
    assert records[0].keys() == records[1].keys()
    for key, value in records[0].items():
        assert bool(torch.isfinite(value).all())
        assert torch.equal(
            value.reshape(-1).view(torch.uint8),
            records[1][key].reshape(-1).view(torch.uint8),
        )


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_global_native_polar_independent_chunked_all_vjps(dtype):
    from benchmarks.cuda.linear.radius_cell_mass import chunked_reference
    from benchmarks.training.cell_mass_mnist import make_radius_cst
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    layer = make_radius_cst(1024, 9, dtype=dtype, radius_execution="global-response4")
    with torch.no_grad():
        layer.atoms.p[:, :2] = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 0],
                [3, 4],
                [-1, 0],
                [0, 1.5],
                [0.3, 0.4],
                [1, 1],
                [-1, 1],
            ],
            dtype=dtype,
        )
        layer.atoms.p[:, 2:] = torch.tensor(
            [
                [0, 1024],
                [-1024, -1e-7],
                [1023.99, 1024.01],
                [4096, -4096],
                [-0.0, 0],
                [2048, -2048],
                [7.3, 8.2],
                [32, 64],
                [128, 256],
            ],
            dtype=dtype,
        )
    hierarchy = hierarchy_from_operator(layer.declaration())
    gen = torch.Generator().manual_seed(2014)
    x = torch.randn(5, 1024, dtype=dtype, generator=gen)
    dy = torch.randn(5, 1024, dtype=dtype, generator=gen)
    expected = chunked_reference(
        copy.deepcopy(layer).double(), x.double(), dy.double(), chunk_atoms=4
    )
    gx, gp = x.cuda().requires_grad_(), layer.atoms.p.detach().cuda().requires_grad_()
    y = ex.radius_linear(hierarchy, gx, gp, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
    grads = torch.autograd.grad(y, (gx, gp), dy.cuda())
    tolerance = (
        dict(atol=3e-5, rtol=3e-4)
        if dtype == torch.float32
        else dict(atol=2e-9, rtol=2e-8)
    )
    for value, target in zip((y, *grads), expected):
        torch.testing.assert_close(value.cpu().double(), target, **tolerance)


@GPU
@pytest.mark.parametrize(
    "empty_atoms,empty_batch", [(True, False), (False, True), (True, True)]
)
def test_global_native_empty_work_matches_python(empty_atoms, empty_batch):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    p = (
        (
            torch.empty(0, 4, dtype=torch.float64)
            if empty_atoms
            else bounded_atoms(plan, torch.float64)
        )
        .cuda()
        .requires_grad_()
    )
    x = torch.zeros(
        0 if empty_batch else 4,
        1024,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    results = []
    for selected in (GLOBAL_ACTIVE_GRAD4, GLOBAL_ACTIVE_GRAD4_NATIVE):
        y = ex.radius_linear(plan, x, p, recipe=selected.recipe)
        results.append((y, *torch.autograd.grad(y, (x, p), torch.ones_like(y))))
    for actual, target in zip(*results):
        assert torch.equal(actual.view(torch.uint8), target.view(torch.uint8))
