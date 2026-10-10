"""Bounded support scratch: chunk tails, decoder snapshots and Graph updates."""

import copy
from dataclasses import replace

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.bounded_recipe import (
    BoundedMatrixProductRecipe,
)
from torchcst._backends.schema import DeviceInfo, ExecutionPlan


def plan(**kwargs):
    return ExecutionPlan(
        "research_profile_product_bounded_matrix",
        "v1",
        BoundedMatrixProductRecipe(**kwargs),
    )


def run(layer, x, family="global", **kwargs):
    assert family == "global"
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=plan(**kwargs)
    )


def test_boundaries_and_recipe_roundtrip():
    layer = scenarios.model(
        torch.tensor([[0.3, 1.5, 2.4, 24]]), "global", n=8192, out=8192
    )
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(7, 8192))),
        device=DeviceInfo("cuda", 0),
    )
    selected = plan()
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(selected)) == selected
    algorithm = REGISTRY.get(
        selected.algorithm_id, revision=selected.algorithm_revision
    )
    assert algorithm.supports(context, selected.recipe).supported
    assert not algorithm.supports(
        replace(context, atom_count=4194305), selected.recipe
    ).supported
    for value in (True, 0, 255, 262145):
        with pytest.raises(ValueError):
            plan(atom_chunk=value)


@pytest.mark.parametrize("n", [2048, 8192])
def test_comparison_snapshots(n):
    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-bounded-{n}-rho3.json",
        "benchmarks/cuda/linear/plans-profile-product-bounded.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    from benchmarks.cuda.linear.profile_product import fixture_operator
    from benchmarks.cuda.linear.run import PlanLinear

    model = PlanLinear(
        torch.tensor([[0.3, 1.5, 2.4, 24]]), fixture_operator(value.case), plan()
    )
    assert model.update_binding is not None
    assert model.local_state is model.live_operator.kernel


@scenarios.GPU
@pytest.mark.parametrize("gemm", ["triton", "torch"])
@pytest.mark.parametrize("preparation", ["full", "support"])
@pytest.mark.parametrize("atoms", [255, 256, 257, 513])
def test_every_chunk_and_tail_matches_full_fp64_oracle(gemm, preparation, atoms):
    torch.manual_seed(41)
    p = torch.randn(atoms, 4) * 0.1 + torch.tensor([0.3, 1.5, 0, 0])
    p[:, 2] = torch.rand(atoms) * 31 + 0.37
    p[:, 3] = torch.rand(atoms) * 63 + 0.37
    layer = scenarios.model(p, "global", device="cuda")
    x = torch.randn(7, 130, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(33, 7, device="cuda").T
    y, dx, dp = scenarios.oracle(layer, x, dy, "global")
    actual = run(layer, x, atom_chunk=256, gemm=gemm, preparation=preparation)
    gx, gp = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in ((actual, y), (gx, dx), (gp, dp)):
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@pytest.fixture
def bounded(monkeypatch):
    monkeypatch.setattr(scenarios, "run", run)


@scenarios.GPU
@pytest.mark.usefixtures("bounded")
@pytest.mark.parametrize(
    "case", ["empty", "singleton", "tiny", "floor", "wide", "precision"]
)
def test_full_norm_floor_and_precision_fallback(case):
    scenarios.test_grouped_global_floor_and_full_support_fallback("global", case)


@scenarios.GPU
@pytest.mark.usefixtures("bounded")
@pytest.mark.parametrize("grad_x,grad_p", [(True, False), (False, True), (True, True)])
def test_empty_atoms_and_gradient_branches(grad_x, grad_p):
    scenarios.test_zero_atoms_and_gradient_branches("global", grad_x, grad_p)


@scenarios.GPU
@pytest.mark.usefixtures("bounded")
def test_twenty_graph_updates_and_moments():
    scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths(
        "global"
    )


@scenarios.GPU
def test_old_forward_retains_all_decoder_scalars_and_lattice():
    torch.manual_seed(41)
    p = torch.tensor([[0.3, 1.5, 2.4, 24]]).repeat(257, 1)
    layer = scenarios.model(p, "global", device="cuda")
    reference = copy.deepcopy(layer)
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn(7, 33, device="cuda")
    truth = reference.operator.apply(xx, algorithm="factored")
    actual = run(layer, x, atom_chunk=256)
    with torch.no_grad():
        layer.atoms.p.add_(0.17)
        for name in (
            "amplitude_max",
            "w_c",
            "kappa",
            "lower_kappa",
            "upper_decay_power",
            "sigma_min_input",
            "sigma_birth_input",
            "sigma_max_input",
            "upper_floor_input",
            "sigma_min_output",
            "sigma_birth_output",
            "sigma_max_output",
            "upper_floor_output",
        ):
            layer.kernel.scalar(name).mul_(0.7)
        layer.chart.axes[0].start.add_(0.25)
        layer.chart.axes[1].start.add_(0.25)
        for axis in layer.chart.axes:
            axis.spacing.fill_(1.25)
    run(layer, x.detach(), atom_chunk=256)
    for value, expected in zip(
        torch.autograd.grad(actual, (x, layer.atoms.p), dy),
        torch.autograd.grad(truth, (xx, reference.atoms.p), dy),
    ):
        torch.testing.assert_close(value, expected, rtol=4e-4, atol=2e-5)
