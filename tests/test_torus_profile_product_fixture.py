"""Square Torus protocol, full-axis oracle and update isolation."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch
from benchmark_database_fixtures import artifact

from benchmarks.cuda.linear import torus_profile_product as fixture
from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from benchmarks.cuda.linear.protocol import (
    TORUS_OPTIMIZER_POLICY,
    TORUS_PRODUCT_ORACLE_SCOPE,
    measurement_operator,
)
from benchmarks.cuda.linear.run import PlanLinear
from benchmarks.cuda.polar_update import optimizer_step
from benchmarks.database.adapters.linear import project
from benchmarks.dispatch.generate import _context
from torchcst import CSTLinear, CSTOptimizer

CATALOG = "benchmarks/cuda/linear/plans-torus-profile-product-chunked.json"


def case(n=1024, sigma=3):
    return load_run(
        f"benchmarks/cuda/linear/cases/torus-profile-product-square-strip-{n}-sigma{sigma}.json",
        CATALOG,
    )


@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("sigma", [3, 8])
def test_square_chart_state_and_frozen_case(n, sigma):
    run = case(n, sigma)
    assert decode_snapshot(run.snapshot()) == run
    op = measurement_operator(asdict(run.case))
    assert op.charts[0].shape == (n, n)
    assert op.charts[0].tile_shape == (64, n)
    assert type(op.charts[0].geometry).__name__ == "TorusGeometrySpec"
    assert op.kernel.revision == 2
    assert run.case.atoms == n * n // 20
    c = replace(run.case, atoms=19)
    p = fixture.initialize(c)
    assert p.shape == (19, 5)
    q = fixture.decode(fixture.fixture_state(c), p)
    torch.testing.assert_close(
        q[:, 1].rsqrt(), torch.full((19,), float(sigma)), rtol=2e-6, atol=2e-6
    )
    with pytest.raises(ValueError, match="fixture size"):
        replace(run.case, size=8192)


@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("route", ["h-saved", "h-recompute", "w-gemm"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_full_axes_and_every_gradient_against_embedded_fibres(n, route, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    run = case(n)
    c = replace(run.case, atoms=4)
    p = fixture.initialize(c).double().to(device)
    model = PlanLinear(p, fixture.fixture_operator(c), run.entry(route).plan)
    torch.manual_seed(41)
    x = torch.randn(3, n, device=device, dtype=torch.float64, requires_grad=True)
    dy = torch.randn(3, n, device=device, dtype=torch.float64)
    expected = fixture.oracle_vjp(
        copy.deepcopy(model.local_state).double(), p, x, dy, model.product_site.chart
    )
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.p), dy)
    for actual, truth in zip((y, dx, dp), expected, strict=True):
        torch.testing.assert_close(actual, truth, rtol=1e-8, atol=1e-10)


def test_large_factor_norm_equals_full_embedded_matrix_norm():
    c = replace(case().case, atoms=4)
    op = fixture.fixture_operator(c)
    p = fixture.initialize(c).double().requires_grad_()
    layer = CSTLinear(
        chart=op.charts[0], atoms=p, kernel=op.kernel, dtype=torch.float64
    )
    u, v, scale = fixture.oracle_factors(layer.kernel, p, layer.chart)
    amp, _ = fixture._polar(layer.kernel, p)
    matrix = u[:, None] * v[None]
    norm = matrix.flatten(0, 1).norm(dim=0).clamp_min(1e-6)
    torch.testing.assert_close(scale, amp / norm, rtol=1e-11, atol=1e-13)


def test_research_update_glue_preserves_public_proposal_and_moments():
    c = replace(case().case, atoms=4)
    op = fixture.fixture_operator(c)
    p = fixture.initialize(c).double()
    model = PlanLinear(p, op, case().entry("h-saved").plan)
    ref = CSTLinear(
        chart=op.charts[0], atoms=p.clone(), kernel=op.kernel, dtype=torch.float64
    )
    base = torch.optim.AdamW(
        ref.parameters(), lr=1e-3, weight_decay=0.02, foreach=False
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.02, foreach=False
    )
    public = CSTOptimizer(base, model=ref)
    gradient = torch.tensor([[0.3, -0.2, 0.5, 2, -1]] * 4, dtype=torch.float64)
    for _ in range(20):
        model.p.grad, ref.atoms.p.grad = gradient.clone(), gradient.clone()
        public.step()
        optimizer_step(model.update_binding, opt, step_size=1e-3, polar_update="torus")
    torch.testing.assert_close(model.p, ref.atoms.p, rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            opt.state[model.p][key], base.state[ref.atoms.p][key], rtol=0, atol=0
        )


def torus_artifact():
    run = case()
    run = replace(run, case=replace(run.case, rounds=3), plans=(run.entry("h-saved"),))
    data = artifact()
    snapshot = run.snapshot()
    sha = hashlib.sha256((json.dumps(snapshot, indent=2) + "\n").encode()).hexdigest()
    data.update(run=snapshot, snapshot_sha256=sha)
    correct, measure, dense = (
        data["records"][0],
        data["records"][2],
        data["records"][-1],
    )
    data["records"] = [correct, measure, dense]
    op = json.loads(json.dumps(asdict(measurement_operator(asdict(run.case)))))
    for row in data["records"]:
        is_dense = row["metadata"]["worker"] == "dense"
        row["metadata"].update(
            case=json.loads(json.dumps(asdict(run.case))),
            input_hashes=run.input_hashes,
            snapshot_sha256=sha,
            polar_update="torus",
            seed=run.case.seed,
            plan_id=None if is_dense else "h-saved",
            plan=None if is_dense else REGISTRY.dump_plan(run.entry("h-saved").plan),
        )
        if row is correct:
            row["result"].update(
                scope=TORUS_PRODUCT_ORACLE_SCOPE,
                polar_update={"max": 1e-6, "rel_l2": 1e-7},
            )
        else:
            row["result"].update(
                operator=op,
                rows=run.case.rows,
                atoms=None if is_dense else run.case.atoms,
                profile=None if is_dense else run.case.profile,
                reference="dense_linear" if is_dense else "h-saved",
                optimizer=asdict(run.case.optimizer),
                optimizer_policy="ordinary AdamW"
                if is_dense
                else TORUS_OPTIMIZER_POLICY,
            )
    return data


def test_torus_projection_and_generation_preserve_geometry_and_update_contract():
    projection = project(torus_artifact())
    assert projection.revision == projection.protocol["revision"] == 8
    row = {
        "case_declaration": projection.case,
        "environment": projection.records[1].environment,
        "adapter_revision": 8,
        "payload": projection.records[1].payload,
        "protocol": projection.protocol,
    }
    ctx = _context(row, "cuda_graph")
    assert ctx.operator.charts[0].shape == (1024, 1024)
    assert ctx.parameter_dim == 5
    row["adapter_revision"] = 7
    with pytest.raises(ValueError, match="revision differs"):
        _context(row, "cuda_graph")


@pytest.mark.parametrize("field", ["policy", "update", "oracle"])
def test_torus_adapter_rejects_euclidean_contracts(field):
    from benchmarks.cuda.linear.protocol import (
        LOCAL_OPTIMIZER_POLICY,
        SQUARE_STRIP_PRODUCT_ORACLE_SCOPE,
    )

    data = torus_artifact()
    if field == "policy":
        data["records"][1]["result"]["optimizer_policy"] = LOCAL_OPTIMIZER_POLICY
    elif field == "update":
        data["records"][0]["metadata"]["polar_update"] = "torch"
    else:
        data["records"][0]["result"]["scope"] = SQUARE_STRIP_PRODUCT_ORACLE_SCOPE
    with pytest.raises(ValueError):
        project(data)
