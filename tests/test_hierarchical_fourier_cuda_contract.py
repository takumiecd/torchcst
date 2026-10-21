"""Research declaration and complete bank coverage without importing Triton."""

import math
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
        {"atom_block": True},
        {"batch_block": 3},
        {"reuse": "stale"},
        {"moment_policy": "stale"},
        {"sample_cache": True},
        {"sample_cache": 512},
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


@pytest.mark.parametrize("modes", [4, 8])
def test_benchmark_contains_nonempty_floor_and_empty_atom(modes):
    from pathlib import Path

    from benchmarks.cuda.linear.hierarchical_fourier import (
        _axis,
        fixture_operator,
        initialize,
    )
    from benchmarks.cuda.linear.manifest import load_run

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    case = load_run(
        root / "cases" / f"hierarchical-fourier-m{modes}-mixed-256-a512.json",
        root / "plans-hierarchical-fourier.json",
    ).case
    p = initialize(case).double()
    u = _axis(p[2:4], fixture_operator(case), 0)
    v = _axis(p[2:4], fixture_operator(case), 1)
    d = u.norm(dim=0) * v.norm(dim=0)
    assert 0 < float(d[0]) < 1e-6
    assert float(d[1]) == 0.0


@pytest.mark.parametrize("modes", [4, 8])
def test_continuous_coefficient_table_value_and_width_derivative(modes):
    from torchcst import FourierPeakSpec
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        coefficient_table,
    )
    from torchcst._backends.torch.profiles.fourier_peak import coefficients

    table, lo = coefficient_table(modes)
    q = torch.linspace(lo + 1e-12, 1 - 1e-12, 10001, dtype=torch.float64)
    t = (2 * q - 1 - lo) / (1 - lo)
    z = torch.zeros(len(q), modes + 1, dtype=torch.float64)
    b1, b2, d1, d2 = z.clone(), z.clone(), z.clone(), z.clone()
    for i in range(len(table) - 1, 0, -1):
        nb = table[i] + 2 * t[:, None] * b1 - b2
        nd = 2 * b1 + 2 * t[:, None] * d1 - d2
        b2, b1 = b1, nb
        d2, d1 = d1, nd
    b = table[0] + t[:, None] * b1 - b2
    db = (b1 + t[:, None] * d1 - d2) * 2 / (1 - lo)
    expected, derivative = coefficients(FourierPeakSpec(modes=modes), q)
    torch.testing.assert_close(b, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(db, derivative, atol=5e-11, rtol=0)


def test_hierarchy_inherits_only_matching_even_frequencies():
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        tables,
    )

    for modes in (4, 8):
        spec = FourierLinearSpec(1024, modes)
        recipe = FourierRecipe(
            minimum_sites=32 if modes == 4 else 64, moment_policy="hierarchical"
        )
        tab = tables(spec, recipe, torch.device("cpu"))
        pairs = bank_families(spec, recipe)
        inherited = 0
        for parent, child in enumerate(tab["children"].tolist()):
            if child < 0:
                continue
            jp, kp, sp = pairs[parent]
            jc, kc, sc = pairs[child]
            assert jp == jc + 1 and kp == 2 * kc and sp == 2 * sc
            for n in range(0, modes + 1, 2):
                torch.testing.assert_close(
                    tab["phi"][parent, :, n],
                    tab["phi"][child, :, n // 2],
                    atol=0,
                    rtol=0,
                )
            inherited += 1
        assert inherited == (6 if modes == 8 else 0)


def test_independent_oracle_cache_uses_exact_inputs_and_verifies_bytes(
    tmp_path, monkeypatch
):
    from benchmarks.cuda.linear.hierarchical_fourier import (
        _ORACLE_CACHE_EVENTS,
        independent_gate,
    )

    monkeypatch.setenv("CST_FOURIER_ORACLE_CACHE", str(tmp_path))
    spec = FourierLinearSpec(64, 4)
    x = torch.randn(2, 64, dtype=torch.float64)
    p = torch.tensor([[0.4, 2.1, 3.2, -3.0]], dtype=torch.float64)
    dy = torch.randn_like(x)
    first = independent_gate(x, p, dy, spec)
    second = independent_gate(x, p, dy, spec)
    assert not _ORACLE_CACHE_EVENTS[-2]["hit"] and _ORACLE_CACHE_EVENTS[-1]["hit"]
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    independent_gate(x + 0.001, p, dy, spec)
    assert not _ORACLE_CACHE_EVENTS[-1]["hit"]
    key = _ORACLE_CACHE_EVENTS[-2]["key"]
    (tmp_path / (key + ".pt")).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        independent_gate(x, p, dy, spec)


def test_sharing_route_declaration_validation_and_fixture_counts():
    from pathlib import Path

    from benchmarks.cuda.linear.fourier_sharing import routing_reference
    from benchmarks.cuda.linear.hierarchical_fourier import initialize
    from benchmarks.cuda.linear.manifest import load_run

    for kw in (
        {"minimum_sharing": True},
        {"minimum_sharing": -1},
        {"minimum_sharing": 1, "strategy": "direct"},
    ):
        with pytest.raises(ValueError):
            FourierRecipe(**kw)
    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    for modes in (4, 8):
        name = f"hierarchical-fourier-sharing-m{modes}-large-cluster-1024-a1024"
        case = load_run(
            root / "cases" / (name + ".json"), root / "plans-hierarchical-fourier.json"
        ).case
        p = initialize(case)
        # No tying of continuous centres or logits, even in the clustered case.
        assert p[4:, 1].unique().numel() == len(p) - 4
        spec = FourierLinearSpec(1024, modes)
        recipe = FourierRecipe(minimum_sites=32, width_cutoff=0.1, minimum_sharing=8)
        counts, routes, selected, _ = routing_reference(p, spec, recipe)
        assert (selected <= counts).all() and routes.any()
        rejected = routing_reference(p, spec, replace(recipe, minimum_sharing=10**6))[1]
        assert not rejected.any()


@pytest.mark.parametrize("modes", [4, 8])
def test_discrete_coefficient_norm_matches_sampled_value_and_gradients(modes):
    from benchmarks.cuda.linear.fourier_sharing import discrete_norm_reference
    from torchcst import FourierPeakSpec
    from torchcst._backends.torch.kernels.hierarchical_fourier import values

    spec = FourierLinearSpec(256, modes)
    p = torch.tensor(
        [
            [0.4, 0.011, 8.371, z]
            for z in (-2.0, -1.0, 0.0, 0.83, 2.0, math.log(6.25 / 3.65))
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = discrete_norm_reference(p, spec)
    width = 0.1 + 9.9 * p[:, 3].sigmoid()
    sites = torch.arange(spec.size, dtype=torch.float64) * 0.1
    expected = []
    for axis in range(2):
        f = values(
            FourierPeakSpec(modes=modes),
            spec.period,
            sites,
            p[:, axis + 1],
            width,
            support="centered",
        )
        total = f.square().sum(0)
        (derivative,) = torch.autograd.grad(total.sum(), p, retain_graph=True)
        sig = p[:, 3].sigmoid()
        expected.append(
            torch.stack(
                (
                    total,
                    derivative[:, axis + 1] / 2,
                    derivative[:, 3] / (2 * 9.9 * sig * (1 - sig)),
                ),
                -1,
            )
        )
    torch.testing.assert_close(actual, torch.stack(expected, 1), atol=3e-10, rtol=3e-10)
