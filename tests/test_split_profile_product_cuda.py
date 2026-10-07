"""Strided/tail reductions, complete VJPs and saved Graph update contracts."""

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe_v4 import (
    SplitMatrixProductRecipe,
    SplitMatrixStripRecipe,
    split_count,
)
from torchcst._backends.schema import ExecutionPlan


def plan(family, large=False, **kw):
    prefix = (
        "research_profile_product"
        if family == "global"
        else "research_strip_profile_product"
    )
    cls = SplitMatrixProductRecipe if family == "global" else SplitMatrixStripRecipe
    return ExecutionPlan(
        f"{prefix}{'_large' if large else ''}_matrix",
        "v2" if large else "v4",
        cls(**kw),
    )


def run(layer, x, family, large=False, **kw):
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=plan(family, large, **kw)
    )


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("split", [1, 4, 8, 16])
def test_roundtrip_and_previous_revision_rejects_split(family, large, split):
    value = plan(family, large, split_k=split)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(value)) == value
    data = REGISTRY.dump_plan(value)
    data["algorithm_revision"] = "v1" if large else "v3"
    with pytest.raises(ValueError):
        REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("invalid", [True, 0, 2, 3, 32, 4.0, None])
def test_split_rejects_other_settings(family, invalid):
    with pytest.raises(ValueError):
        plan(family, split_k=invalid)


@pytest.mark.parametrize("family", ["global", "strip"])
def test_split_rejects_torch_engine(family):
    with pytest.raises(ValueError):
        plan(family, gemm="torch")


@pytest.mark.parametrize(
    "k,requested,expected",
    [
        (1, 16, 1),
        (32, 16, 1),
        (33, 16, 2),
        (65, 16, 4),
        (129, 16, 8),
        (257, 16, 16),
        (8192, 4, 4),
        (8192, 1, 1),
    ],
)
def test_reduction_caps_at_available_blocks(k, requested, expected):
    assert split_count(k, requested) == expected


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("large,size", [(False, 1024), (True, 2048), (True, 8192)])
@pytest.mark.parametrize("rho", [3, 8])
def test_split_case_snapshots_and_live_optimizer_binding(family, large, size, rho):
    from dataclasses import asdict

    from benchmarks.cuda.linear.protocol import measurement_operator
    from benchmarks.cuda.linear.run import PlanLinear

    tag = "large-" if large else ""
    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-{tag}{family}-{size}-rho{rho}-split-k.json",
        f"benchmarks/cuda/linear/plans-profile-product-{tag}{family}-split-k.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    layer = PlanLinear(
        torch.tensor([[0.3, 1.5, 2.4, 24]]),
        measurement_operator(asdict(value.case)),
        value.entry("split8").plan,
    )
    assert layer.update_binding is not None
    assert layer.local_state is layer.live_operator.kernel


@scenarios.GPU
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 1),
        (7, 33, 17),
        (32, 65, 33),
        (64, 3, 65),
        (7, 33, 129),
        (32, 65, 257),
        (1, 33, 1025),
    ],
)
@pytest.mark.parametrize("split", [4, 16])
@pytest.mark.parametrize("transposed", [False, True])
def test_native_split_all_tails_and_strides(m, n, k, split, transposed):
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.executor import (
        _matmul,
    )

    torch.manual_seed(41)
    if transposed:
        left = torch.randn(k * 2, m * 2, device="cuda")[::2, ::2].T
        right = torch.randn(n * 2, k * 2, device="cuda")[::2, ::2].T
    else:
        left = torch.randn(m * 2, k * 2, device="cuda")[::2, ::2]
        right = torch.randn(k * 2, n * 2, device="cuda")[::2, ::2]
    result = _matmul(left, right, SplitMatrixProductRecipe(split_k=split))
    torch.testing.assert_close(
        result.double(), left.double() @ right.double(), rtol=4e-4, atol=2e-5
    )


@pytest.fixture(params=[(False, 4), (False, 16), (True, 8)])
def split_route(request, monkeypatch):
    large, split = request.param
    original = scenarios.model

    def model(p, family, **kw):
        if large:
            kw.setdefault("n", 2049 if family == "global" else 8191)
            kw.setdefault("out", 2051 if family == "global" else 65)
        return original(p, family, **kw)

    def execute(layer, x, family, **kw):
        return run(layer, x, family, large, split_k=split, **kw)

    monkeypatch.setattr(scenarios, "model", model)
    monkeypatch.setattr(scenarios, "run", execute)


@pytest.mark.usefixtures("split_route")
class TestSplitOperator:
    test_all_sites = staticmethod(
        scenarios.test_grouped_all_sites_and_canonical_gradients
    )
    test_floor = staticmethod(
        scenarios.test_grouped_global_floor_and_full_support_fallback
    )
    test_pitch = staticmethod(
        scenarios.test_strip_pitch_boundaries_gaps_and_partial_tile_have_exact_full_norm
    )
    test_snapshots = staticmethod(
        scenarios.test_old_forward_snapshots_keep_pitch_amplitude_and_parameters
    )
    test_gradients = staticmethod(scenarios.test_zero_atoms_and_gradient_branches)
    test_updates = staticmethod(
        scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths
    )
    test_live_pitch = staticmethod(
        scenarios.test_captured_strip_reads_changed_pitch_and_spacing_fallback
    )


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
def test_large_split_complete_chart_with_tails(family):
    torch.manual_seed(41)
    n, out = (2049, 2051) if family == "global" else (8191, 65)
    p = torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]], device="cuda")
    layer = scenarios.model(p, family, n=n, out=out, device="cuda")
    x = torch.randn(7, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(out, 7, device="cuda").T
    truth = scenarios.oracle(layer, x, dy, family)
    y = run(layer, x, family, True, split_k=16)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for value, reference in zip((y, dx, dp), truth, strict=True):
        torch.testing.assert_close(value.double(), reference, rtol=4e-4, atol=2e-5)
