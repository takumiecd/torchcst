"""Independent physical and output-edge ownership contracts for site gather."""

import os
import subprocess
import sys
from collections import Counter
from dataclasses import fields, replace

import pytest
import test_sphere_direct as direct
import test_sphere_direct_recompute as recompute
import torch

from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_algorithm import (
    SphereSiteGatherAlgorithm,
    SphereSiteGatherRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import context_from_tensors

cuda = direct.cuda
OPTIONS = ((1, True), (4, True))


@pytest.fixture(autouse=True)
def site_binding(monkeypatch):
    def recipe(options=(4, True), **kwargs):
        return SphereSiteGatherRecipe(site_group=options[0], **kwargs)

    monkeypatch.setattr(direct, "SphereDirectAlgorithm", SphereSiteGatherAlgorithm)
    monkeypatch.setattr(direct, "recipe", recipe)
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def test_site_recipe_strict_metadata_and_int32_edge_capacity_guard():
    algorithm, registry = SphereSiteGatherAlgorithm(), Registry()
    registry.register(algorithm)
    for group in (1, 4):
        recipe = SphereSiteGatherRecipe(site_group=group)
        plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for field in fields(SphereSiteGatherRecipe()):
        if field.name == "merge_output_vjp":
            bad = False
        elif field.name == "site_group":
            bad = 2
        else:
            bad = 1
        with pytest.raises(ValueError):
            SphereSiteGatherRecipe(**{field.name: bad})
        with pytest.raises(ValueError):
            SphereSiteGatherRecipe(
                **{field.name: 1 if field.name == "merge_output_vjp" else True}
            )
    with pytest.raises(TypeError):
        algorithm.validate_recipe(object())
    model, x, *_ = direct.fixture(17, 3.0, atoms=9)
    context = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    recipe = SphereSiteGatherRecipe()
    assert algorithm.supports(context, recipe).supported
    limit = (2**31 - 1) // 64
    assert algorithm.supports(replace(context, atom_count=limit), recipe).supported
    assert not algorithm.supports(
        replace(context, atom_count=limit + 1), recipe
    ).supported
    # Meta tensors exercise the allocation guard without allocating a large array
    # or importing/launching Triton. The rejection precedes all geometry work.
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_executor import (
        _Gather,
        build_output_csr,
    )

    index = torch.empty((limit + 1, 64), device="meta", dtype=torch.int16)
    count = torch.empty(limit + 1, device="meta", dtype=torch.int32)
    with pytest.raises(ValueError, match="int32"):
        build_output_csr(index, count, 17)
    p = torch.empty((limit + 1, 6), device="meta")
    with pytest.raises(ValueError, match="int32"):
        _Gather.forward(None, x.to("meta"), p, None, None, recipe, True)
    for bad in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, deterministic=True),
        replace(context, parameter_dim=5),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
    ):
        assert not algorithm.supports(bad, recipe).supported


def test_site_declaration_does_not_load_gpu_execution():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_algorithm import SphereSiteGatherAlgorithm, SphereSiteGatherRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
a = SphereSiteGatherAlgorithm()
r = Registry()
r.register(a)
p = ExecutionPlan(a.id, a.revision, SphereSiteGatherRecipe())
assert r.loads_plan(r.dumps_plan(p)) == p
assert 'triton' not in sys.modules
assert not any(n.endswith(('.site_gather_executor', '.site_gather_kernels', '.recompute_prepare')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@cuda
@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_site_gather_all_six_and_independent_requested_gradients(
    options, need_x, need_p
):
    direct.test_full_fp64_all_six_and_requested_gradients(
        options, 17, 1.0, 9, need_x, need_p
    )


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 3, 4, 5])
def test_site_gather_empty_and_atom_group_tails(atoms):
    direct.test_empty_atoms_and_group_tails(atoms)


@cuda
@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("gap", [0.0, 2**-9, 1.0])
def test_site_gather_empty_subfloor_and_above_floor(options, gap):
    direct.test_singleton_empty_and_both_norm_floor_branches(options, gap)


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_site_gather_rectangular_strides_and_asymmetric_geometry(options):
    direct.test_rectangular_strides_asymmetric_widths_and_antipodes(options)


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_site_gather_retained_forward_and_new_live_geometry(options):
    direct.test_retained_forward_owns_live_geometry_and_scalars(options)


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_site_gather_graph_reads_live_scalar_and_geometry_changes(options):
    direct.test_graph_replays_live_widths_geometry_and_full_gradients(options)


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_site_gather_twenty_public_updates_and_exact_clock(options):
    direct.test_twenty_public_updates_keep_parameters_moments_and_exact_clock(options)


@cuda
def test_site_gather_twenty_complete_graph_updates_match_public_reference():
    recompute.test_recomputed_phi_twenty_complete_graph_updates_match_public_reference()


@cuda
@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_site_gather_mixed_complete_overflow_and_signed_amplitudes(
    options, need_x, need_p
):
    direct.test_one_group_mixes_all_complete_overflow_paths_and_signed_amplitudes(
        options, need_x, need_p
    )


@cuda
def test_output_csr_edge_bijection_high_degree_empty_rows_and_overflow_exclusion():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_executor import (
        build_output_csr,
    )

    index = torch.full((81, 64), -123, dtype=torch.int16, device="cuda")
    count = torch.full((81,), 2, dtype=torch.int32, device="cuda")
    count[0], count[1], count[2], count[3], count[-1] = 0, 65, 64, 1, 0
    index[2] = torch.arange(64, dtype=torch.int16, device="cuda")
    index[3, 0] = 66
    index[4:80, 0], index[4:80, 1] = 0, 66
    rowptr, edges = build_output_csr(index, count, 67, support_capacity=64)
    torch.cuda.synchronize()
    assert rowptr.dtype == edges.dtype == torch.int32
    assert rowptr.shape == (68,) and edges.shape == (81 * 64,)
    expected = [[] for _ in range(67)]
    for atom, degree in enumerate(count.cpu().tolist()):
        if degree <= 64:
            for site in index[atom, :degree].cpu().tolist():
                expected[site].append(atom)
    prefix = [0]
    for row in expected:
        prefix.append(prefix[-1] + len(row))
    assert rowptr.cpu().tolist() == prefix
    assert len(expected[0]) == len(expected[66]) == 77
    assert len(expected[0]) > index.shape[1]
    assert expected[64] == expected[65] == []
    assert prefix[-1] == 217
    for site, row in enumerate(expected):
        actual = edges[prefix[site] : prefix[site + 1]].cpu().tolist()
        assert Counter(actual) == Counter(row)
        assert 1 not in actual  # The complete-axis fallback owns this atom.


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_site_owner_contracts_more_than_capacity_atoms_per_row(options):
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_prepare import (
        prepare_ids,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_executor import (
        build_output_csr,
    )

    model, x, _, dy = direct.fixture(17, 16.0, batch=3, atoms=81)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    prepared = prepare_ids(
        x, model.atoms.p, model.kernel, model.cst_charts(), direct.recipe(options)
    )
    assert all((side[7] == 17).all() for side in prepared[5])
    rowptr, _ = build_output_csr(prepared[5][1][4], prepared[5][1][7], 17)
    assert torch.equal(rowptr[1:] - rowptr[:-1], torch.full_like(rowptr[1:], 81))
    y = direct.binding(model, options)(x)
    direct.gate(
        (y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 5])
def test_output_csr_empty_edges_and_small_atom_tails(atoms):
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.site_gather_executor import (
        build_output_csr,
    )

    index = torch.full((atoms, 64), -1, dtype=torch.int16, device="cuda")
    count = torch.zeros(atoms, dtype=torch.int32, device="cuda")
    rowptr, edges = build_output_csr(index, count, 17, support_capacity=64)
    assert rowptr.shape == (18,) and edges.shape == (atoms * 64,)
    assert torch.count_nonzero(rowptr) == 0


@cuda
def test_site_gather_has_bounded_csr_no_phi_and_saves_independent_h(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from torchcst._backends.cuda.algorithms.linear.sphere_polar import (
        site_gather_executor,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_prepare import (
        prepare_ids,
    )

    model, x, _, dy = direct.fixture(17, 1.0, batch=3, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    call = direct.binding(model)
    warm = call(x)
    torch.autograd.grad(warm, (x, model.atoms.p), dy)
    scratch, saved, allocations = [], [], []
    original_builder = site_gather_executor.build_output_csr

    def observed_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        scratch.extend(result)
        assert result[0].shape == (18,) and result[1].shape == (19 * 64,)
        return result

    monkeypatch.setattr(site_gather_executor, "build_output_csr", observed_builder)

    def save(value):
        saved.append(value)
        return value

    class BoundedAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))

            def inspect(value):
                if isinstance(value, torch.Tensor):
                    allocations.append((tuple(value.shape), value.dtype))
                    assert not (
                        value.is_floating_point() and tuple(value.shape) == (19, 64)
                    ), "Phi allocation"
                    assert tuple(value.shape) != (17, 17), "W/dW allocation"
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect(item)

            inspect(result)
            return result

    with (
        BoundedAllocations(),
        torch.autograd.graph.saved_tensors_hooks(save, lambda t: t),
    ):
        y = call(x)
    assert len(scratch) == 2
    scratch_ids = {t.data_ptr() for t in scratch}
    assert not scratch_ids.intersection(t.data_ptr() for t in saved)
    assert ((19, 64), torch.int16) in allocations
    assert ((19 * 64,), torch.int32) in allocations
    _, _, _, _, _, sides = prepare_ids(
        x, model.atoms.p, model.kernel, model.cst_charts(), direct.recipe()
    )
    sites, centers, _, precision, *_ = sides[0]
    _, raw = recompute.full_axis_raw(sites, centers, precision)
    phi = raw.double() / raw.double().square().sum(-1).sqrt().clamp_min(1e-6)[:, None]
    expected_h = phi @ x.detach().double().T
    matching_h = [
        t
        for t in saved
        if t.shape == expected_h.shape
        and direct.error(t, expected_h)["max_abs"] <= 4e-4
        and direct.error(t, expected_h)["relative_l2"] <= 4e-4
    ]
    assert len(matching_h) == 1
    h = matching_h[0]
    prior_h = h.detach().clone()
    newer = call(x * 0.7)
    torch.autograd.grad(newer, model.atoms.p, dy)
    torch.testing.assert_close(h, prior_h, atol=0, rtol=0)
    direct.gate(
        (y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )


@cuda
@pytest.mark.parametrize("options", OPTIONS)
def test_l4_compiled_site_owners_and_csr_use_cuda_core_instructions(
    options, monkeypatch, tmp_path
):
    import hashlib
    import json
    from pathlib import Path

    from torchcst._backends.cuda.algorithms.linear.sphere_polar import (
        direct_kernels,
        site_gather_kernels,
    )

    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        pytest.skip("compiler resource diagnostic requires NVIDIA L4")
    captured = {}

    class CaptureCompiled:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                compiled = launch(*args, **kwargs)
                captured[self.name] = compiled
                return compiled

            return run

    expected = {
        "count_edges",
        "prefix_rows",
        "scatter_edges",
        "h_forward",
        "owner_output",
        "overflow_output",
        "backward",
    }
    for name in expected:
        module = direct_kernels if name == "backward" else site_gather_kernels
        monkeypatch.setattr(module, name, CaptureCompiled(getattr(module, name), name))
    model, x, _, dy = direct.fixture(2048, 3.0, batch=32, atoms=int(0.05 * 2048**2))
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    y = direct.binding(model, options)(x)
    gradients = torch.autograd.grad(y, (x, model.atoms.p), dy)
    torch.cuda.synchronize()
    assert all(torch.isfinite(value).all() for value in (y, *gradients))
    folder = Path(os.environ.get("CST_JOB_OUTPUT", str(tmp_path)))
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"compiler-site-gather-r{options[0]}"
    records = {}
    for name, compiled in captured.items():
        ptx = compiled.asm["ptx"]
        path = folder / f"{stem}-{name}.ptx"
        path.write_text(ptx)
        records[name] = {
            "n_regs": compiled.n_regs,
            "n_spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared,
            "ptx_file": path.name,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            **direct.ptx_arithmetic_report(ptx),
        }
    report = {
        "scope": "separate compiled-resource diagnostic; not a timing/adoption gate",
        "algorithm_id": SphereSiteGatherAlgorithm().id,
        "gpu": torch.cuda.get_device_name(0),
        "size": 2048,
        "batch": 32,
        "atoms": int(0.05 * 2048**2),
        "recipe": vars(direct.recipe(options)),
        "kernels": records,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (folder / f"{stem}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    assert set(records) == expected
    for name, record in records.items():
        assert (
            record["n_regs"] > 0
            and record["n_spills"] >= 0
            and record["shared_bytes"] >= 0
        )
        assert not record["mma_present"] and not record["tf32_present"]
        if name in {"h_forward", "owner_output", "overflow_output", "backward"}:
            assert all(record["fp32_instruction_samples"].values())
