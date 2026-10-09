"""Independent Phi-free contracts, sharing the original physical test fixtures."""

import os
import subprocess
import sys

import pytest
import test_sphere_direct as direct
import torch

from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_algorithm import (
    SphereDirectRecomputeAlgorithm,
    SphereDirectRecomputeRecipe,
)

cuda = direct.cuda
VARIANTS = direct.VARIANTS


@pytest.fixture(autouse=True)
def recompute_binding(monkeypatch):
    # Reuse independently grounded physical witnesses, changing only the Plan.
    monkeypatch.setattr(direct, "SphereDirectAlgorithm", SphereDirectRecomputeAlgorithm)
    monkeypatch.setattr(direct, "SphereDirectRecipe", SphereDirectRecomputeRecipe)
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def test_recompute_recipe_strict_metadata_and_roundtrip():
    direct.test_recipe_roundtrip_and_strict_metadata()
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.direct_algorithm import (
        SphereDirectRecipe,
    )

    with pytest.raises(TypeError):
        SphereDirectRecomputeAlgorithm().validate_recipe(SphereDirectRecipe())


def test_recompute_declaration_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_algorithm import SphereDirectRecomputeAlgorithm, SphereDirectRecomputeRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
a = SphereDirectRecomputeAlgorithm()
r = Registry()
r.register(a)
p = ExecutionPlan(a.id, a.revision, SphereDirectRecomputeRecipe())
assert r.loads_plan(r.dumps_plan(p)) == p
assert 'triton' not in sys.modules
assert not any(n.endswith(('.direct_executor', '.direct_kernels', '.recompute_prepare', '.recompute_prepare_kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_recomputed_phi_full_fp64_and_requested_gradients(options, need_x, need_p):
    direct.test_full_fp64_all_six_and_requested_gradients(
        options, 17, 1.0, 9, need_x, need_p
    )


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 3, 4, 5])
def test_recomputed_phi_empty_and_atom_group_tails(atoms):
    direct.test_empty_atoms_and_group_tails(atoms)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("gap", [0.0, 2**-9, 2**-8, 1.0])
def test_recomputed_phi_empty_subfloor_and_above_floor(options, gap):
    direct.test_singleton_empty_and_both_norm_floor_branches(options, gap)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_recomputed_phi_rectangular_strides_and_asymmetric_geometry(options):
    direct.test_rectangular_strides_asymmetric_widths_and_antipodes(options)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_recomputed_phi_retains_geometry_norms_and_live_scalar_snapshots(options):
    direct.test_retained_forward_owns_live_geometry_and_scalars(options)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_recomputed_phi_graph_replays_live_geometry_and_widths(options):
    direct.test_graph_replays_live_widths_geometry_and_full_gradients(options)


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_recomputed_phi_twenty_public_updates_and_exact_optimizer_clock(options):
    direct.test_twenty_public_updates_keep_parameters_moments_and_exact_clock(options)


@cuda
def test_recomputed_phi_twenty_complete_graph_updates_match_public_reference():
    import copy

    from benchmarks.cuda.linear.scaling_comparison import ProposalUpdate, capture
    from torchcst import CSTOptimizer

    model, x, target, dy = direct.fixture(32, 3.0, batch=3, atoms=9)
    model, x, target, dy = (
        model.cuda(),
        x.cuda().requires_grad_(),
        target.cuda(),
        dy.cuda(),
    )
    call = direct.binding(model)
    actual_update = ProposalUpdate(model, "sphere", capturable=True)

    def step():
        actual_update.opt.zero_grad(set_to_none=True)
        x.grad = None
        y = call(x)
        loss = (y - target).square().mean()
        loss.backward()
        actual_update()
        return y, loss

    graph, _ = capture(step)
    initial_step = int(actual_update.opt.state[model.atoms.p]["step"])
    assert initial_step == 3
    reference = copy.deepcopy(model)
    base = torch.optim.AdamW(
        reference.parameters(), lr=1e-4, weight_decay=0.01, fused=True
    )
    base.load_state_dict(copy.deepcopy(actual_update.opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    public = CSTOptimizer(base, model=reference)
    xx = x.detach().clone().requires_grad_()
    for index in range(20):
        public.zero_grad(set_to_none=True)
        xx.grad = None
        (reference(xx) - target).square().mean().backward()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        for name, aa, bb, tolerance in (
            ("parameters", model.atoms.p, reference.atoms.p, 2e-6),
            (
                "exp_avg",
                actual_update.opt.state[model.atoms.p]["exp_avg"],
                base.state[reference.atoms.p]["exp_avg"],
                4e-4,
            ),
            (
                "exp_avg_sq",
                actual_update.opt.state[model.atoms.p]["exp_avg_sq"],
                base.state[reference.atoms.p]["exp_avg_sq"],
                4e-4,
            ),
        ):
            assert torch.isfinite(aa).all() and torch.isfinite(bb).all(), name
            metrics = direct.error(aa, bb)
            assert metrics["max_abs"] <= tolerance, (name, metrics)
        assert (
            int(actual_update.opt.state[model.atoms.p]["step"])
            == initial_step + index + 1
        )
        assert int(base.state[reference.atoms.p]["step"]) == initial_step + index + 1
    y = call(x)
    direct.gate(
        (y, *torch.autograd.grad(y, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )


@cuda
@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_recomputed_phi_complete_mixed_overflow_and_signed_amplitudes(
    options, need_x, need_p
):
    direct.test_one_group_mixes_all_complete_overflow_paths_and_signed_amplitudes(
        options, need_x, need_p
    )


def full_axis_raw(sites, centers, precision):
    """Torch arithmetic over every site; no packed/kernel implementation calls."""
    delta = sites[None, :, :] - centers[:, None, :]
    square = delta.square()
    distance = (square[:, :, 0] + square[:, :, 1]) + square[:, :, 2]
    gap = (1 - distance * precision[:, None]).clamp_min(0)
    return gap, gap * gap * gap


@cuda
@pytest.mark.parametrize("size,sigma", [(17, 1.0), (128, 16.0)])
def test_id_only_pack_has_complete_chart_counts_ids_and_norms(size, sigma):
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_prepare import (
        prepare_ids,
    )

    model, x, *_ = direct.fixture(size, sigma, batch=3, atoms=9)
    model, x = model.cuda(), x.cuda()
    prepared = prepare_ids(
        x, model.atoms.p, model.kernel, model.cst_charts(), direct.recipe()
    )
    for sites, centers, _, precision, index, phi, norm, count in prepared[5]:
        assert phi.shape == (0,) and phi.untyped_storage().nbytes() == 0
        assert index.dtype == torch.int16 and index.shape == (9, 64)
        gap, raw = full_axis_raw(sites, centers, precision)
        expected_count = (gap > 0).sum(-1)
        torch.testing.assert_close(count.long(), expected_count, atol=0, rtol=0)
        # FP64 reduction of the independently evaluated full-chart FP32 raw.
        expected_norm = raw.double().square().sum(-1).sqrt()
        direct.gate((norm,), (expected_norm,))
        for a in range(len(count)):
            if int(count[a]) <= 64:
                expected_id = torch.nonzero(gap[a] > 0).flatten()
                torch.testing.assert_close(
                    index[a, : len(expected_id)].long(), expected_id, atol=0, rtol=0
                )
    if sigma == 16:
        assert all((side[7] > 64).all() for side in prepared[5])


@cuda
def test_id_only_pack_retains_positive_support_when_squared_raw_flushes():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_prepare_kernels import (
        pack_ids,
    )

    sites = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda")
    centers = torch.zeros(1, 3, device="cuda")
    precision = torch.nextafter(
        torch.ones(1, device="cuda"), torch.zeros(1, device="cuda")
    )
    index = torch.full((1, 64), -1, dtype=torch.int16, device="cuda")
    norm = torch.full((1,), -1.0, device="cuda")
    count = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    pack_ids[(1,)](
        sites,
        centers,
        precision,
        index,
        norm,
        count,
        2,
        64,
        2,
        num_warps=4,
        enable_fp_fusion=False,
    )
    gap, raw = full_axis_raw(sites, centers, precision)
    assert float(gap[0, 0]) == 2**-24 and float(raw[0, 0]) > 0
    assert int(count[0]) == 1 and int(index[0, 0]) == 0
    # CUDA FTZ can erase raw^2, but it must not erase the positive-gap support.
    assert float(norm[0]) == 0.0


@cuda
def test_no_float_atom_by_capacity_profile_allocation(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from torchcst._backends.cuda.algorithms.linear.sphere_polar import recompute_prepare
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.recompute_prepare import (
        prepare_ids,
    )

    model, x, _, dy = direct.fixture(17, 1.0, batch=3, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    call = direct.binding(model)
    y = call(x)
    torch.autograd.grad(y, (x, model.atoms.p), dy)
    calls, allocations = [], []

    def observed(*args, **kwargs):
        result = prepare_ids(*args, **kwargs)
        assert all(side[5].numel() == 0 for side in result[5])
        calls.append(True)
        return result

    monkeypatch.setattr(recompute_prepare, "prepare_ids", observed)

    class NoPhi(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))

            def inspect(value):
                if isinstance(value, torch.Tensor):
                    allocations.append((tuple(value.shape), value.dtype))
                    assert not (
                        value.is_floating_point() and tuple(value.shape) == (19, 64)
                    ), "saved Phi-sized float allocation"
                    assert tuple(value.shape) != (17, 17), "W/dW allocation"
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect(item)

            inspect(result)
            return result

    with NoPhi():
        y = call(x)
        actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    assert calls == [True]
    assert ((19, 64), torch.int16) in allocations
    assert ((19, 3), torch.float32) in allocations  # H and centres are permitted.
    direct.gate(actual, direct.oracle_vjp(model, x, dy))


@cuda
def test_recomputed_block_widens_maximum_signed_id_and_never_reads_phi():
    import triton
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.direct_kernels import (
        _block,
    )

    @triton.jit
    def probe(S, Q, P, I, F, N, C, Out):
        a = tl.arange(0, 4)
        live = a < 1
        count = tl.load(C + a, live, 0)
        idx, _, phi, gap, _, _, _, _ = _block(
            S, Q, P, I, F, N, a, live, count, 0, 32768, 64, 16, False, 1e-6, True
        )
        t = tl.arange(0, 16)
        tl.store(Out + a[:, None] * 16 + t[None, :], idx)
        tl.store(Out + 64 + a[:, None] * 16 + t[None, :], phi * gap)

    sites = torch.zeros(32768, 3, device="cuda")
    sites[-1, 0] = 2.0
    centers = torch.tensor([[2.0, 0.0, 0.0]], device="cuda")
    precision = torch.ones(1, device="cuda")
    index = torch.zeros(1, 64, dtype=torch.int16, device="cuda")
    index[0, 0] = 32767
    empty_phi = torch.empty(0, device="cuda")
    norm = torch.full((1,), 2.0, device="cuda")
    count = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(128, device="cuda")
    probe[(1,)](
        sites,
        centers,
        precision,
        index,
        empty_phi,
        norm,
        count,
        out,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert float(out[0]) == 32767 and float(out[64]) == 0.5


@cuda
@pytest.mark.parametrize("options", VARIANTS)
def test_l4_compiled_recompute_uses_fp32_cuda_core_instructions(
    options, monkeypatch, tmp_path
):
    import json
    from pathlib import Path

    # Keep the False-path diagnostic artifacts distinct if both suites run.
    folder = (
        Path(os.environ.get("CST_JOB_OUTPUT", str(tmp_path))) / "compiler-recompute"
    )
    monkeypatch.setenv("CST_JOB_OUTPUT", str(folder))
    direct.test_l4_compiled_direct_uses_fp32_cuda_core_instructions(
        options, monkeypatch, tmp_path
    )
    group, merge = options
    report_path = folder / f"compiler-direct-g{group}-merge{int(merge)}.json"
    report = json.loads(report_path.read_text())
    report["algorithm_id"] = SphereDirectRecomputeAlgorithm().id
    report["recompute_phi"] = True
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
