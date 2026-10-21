"""Research declaration and complete bank coverage without importing Triton."""

from dataclasses import replace

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.algorithm import (
    FourierContext,
    HierarchicalFourierAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    FourierLinearSpec,
    FourierRecipe,
    torch_reference,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
    bank_families,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst._backends.torch.kernels.hierarchical_fourier import components


def test_metadata_roundtrip_and_rejections():
    alg = HierarchicalFourierAlgorithm()
    registry = Registry()
    registry.register(alg)
    plan = ExecutionPlan(alg.id, alg.revision, FourierRecipe())
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    ctx = FourierContext(FourierLinearSpec(1024), "cuda", torch.float32, 32, 512)
    assert alg.supports(ctx, plan.recipe).supported
    assert not alg.supports(replace(ctx, device_type="cpu"), plan.recipe).supported
    assert not alg.supports(replace(ctx, parameter_dim=5), plan.recipe).supported
    for kw in (
        {"strategy": "invalid"},
        {"atom_chunk": True},
        {"minimum_sites": 2},
        {"width_cutoff": float("nan")},
    ):
        with pytest.raises(ValueError):
            FourierRecipe(**kw)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("minimum_sites", [32, 64, 128])
def test_no_missing_live_period_bank(modes, minimum_sites):
    spec = FourierLinearSpec(1024, modes)
    recipe = FourierRecipe(minimum_sites=minimum_sites)
    pairs = {(j, k) for j, k, _ in bank_families(spec, recipe)}
    widths = torch.linspace(0.1, 10, 10001, dtype=torch.float64)
    _L, T, w, _dw, k, j = components(spec.period, widths)
    required = (
        (w > 1e-13)
        & (T / spec.spacing >= minimum_sites - 1e-10)
        & (T / spec.spacing <= spec.size + 1e-10)
        & (widths[:, None] >= recipe.width_cutoff)
    )
    assert {(int(jj), int(kk)) for jj, kk in zip(j[required], k[required])} <= pairs
    assert all(
        round(spec.period.minimum_tile_width * 2**jj / spec.spacing) >= minimum_sites
        for jj, _ in pairs
    )


@pytest.mark.parametrize("modes", [4, 8])
def test_reference_has_live_width_and_zero_amplitude_derivative(modes):
    spec = FourierLinearSpec(64, modes)
    x = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
    p = torch.tensor(
        [[0.0, 2.137, 3.253, -3.0], [0.6, 1.75, 4.3, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch_reference(x, p, spec)
    dx, dp = torch.autograd.grad(y.sum(), (x, p))
    assert torch.isfinite(dx).all() and torch.isfinite(dp).all()
    assert dp[0, 0].abs() > 1e-8
    assert dp[1, 3].abs() > 1e-8


@pytest.mark.parametrize("modes", [4, 8])
def test_independent_fixture_oracle_matches_torch_reference(modes):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate

    spec = FourierLinearSpec(64, modes)
    gen = torch.Generator().manual_seed(41)
    p = torch.randn(12, 4, generator=gen, dtype=torch.float64)
    p[:, 1:3] = torch.rand(12, 2, generator=gen, dtype=torch.float64) * 6.3
    p[0, 1:3] = -0.01
    p[1, 1:3] = -20
    p[2, 0] = 0
    p.requires_grad_()
    x = torch.randn(3, 64, generator=gen, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    y = torch_reference(x, p, spec)
    dx, dp = torch.autograd.grad(y, (x, p), dy)
    yy, xx, pp = independent_gate(x, p, dy, spec, chunk=4)
    for a, b in ((y, yy), (dx, xx), (dp, pp)):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)


def test_benchmark_catalog_and_case_snapshot_roundtrip():
    from dataclasses import asdict
    from pathlib import Path

    from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
    from benchmarks.cuda.linear.protocol import measurement_operator

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    for path in sorted((root / "cases").glob("hierarchical-fourier-*.json")):
        run = load_run(path, root / "plans-hierarchical-fourier.json")
        assert decode_snapshot(run.snapshot()) == run
        assert measurement_operator(asdict(run.case)).modes == int(run.case.profile[1])
