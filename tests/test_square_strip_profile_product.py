"""N by N Strip fixture, versioned capabilities, exact full-site oracle and Graphs."""

import copy
from dataclasses import asdict, replace

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear import strip_profile_product as fixture
from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from benchmarks.cuda.linear.protocol import measurement_operator
from torchcst import CSTLinear, Dispatcher, LinearInputs
from torchcst._backends.schema import DeviceInfo

CATALOG = "benchmarks/cuda/linear/plans-profile-product-square-strip.json"


def case(n=1024, rho=3):
    return load_run(
        f"benchmarks/cuda/linear/cases/profile-product-square-strip-{n}-rho{rho}.json",
        CATALOG,
    )


@pytest.mark.parametrize("n", [1024, 2048, 8192])
@pytest.mark.parametrize("rho", [3, 8])
def test_square_declaration_and_old_64_output_fixture(n, rho):
    run = case(n, rho)
    assert decode_snapshot(run.snapshot()) == run
    op = measurement_operator(asdict(run.case))
    assert op.charts[0].shape == (n, n)
    assert op.charts[0].tile_shape == (n, 64)
    assert run.case.atoms == n * n // 20
    small = replace(run.case, atoms=19)
    p = fixture.initialize(small)
    assert p[:, 2].max() > 128
    assert p[:, 2].max() < n
    old = replace(small, fixture="polar_profile_product_strip")
    assert fixture.fixture_operator(old).charts[0].shape == (64, n)
    assert fixture.initialize(old)[:, 2].max() < 64


@pytest.mark.parametrize("alias", ["prepared", "split1", "split8", "torch-g8-p8"])
def test_versioned_output_limits(alias):
    run = case(8192)
    layer = scenarios.model(
        torch.tensor([[0.3, 1.5, 7000.37, 24.37]]), "strip", n=8192, out=8192
    )
    ctx = replace(
        layer.build_context(LinearInputs(torch.randn(1, 8192))),
        device=DeviceInfo("cuda", 0),
    )
    selected = run.entry(alias).plan
    alg = REGISTRY.validate_plan(selected)
    assert alg.supports(ctx, selected.recipe).supported
    assert not alg.supports(replace(ctx, atom_count=4194305), selected.recipe).supported
    old_id = selected.algorithm_id.replace(
        "research_square_strip_profile_product", "research_strip_profile_product_large"
    )
    old = REGISTRY.get(old_id, revision=selected.algorithm_revision)
    assert not old.supports(ctx, selected.recipe).supported
    too_large = scenarios.model(layer.atoms.p.detach(), "strip", n=8192, out=8193)
    bad = replace(
        too_large.build_context(LinearInputs(torch.randn(1, 8192))),
        device=DeviceInfo("cuda", 0),
    )
    assert not alg.supports(bad, selected.recipe).supported


@pytest.mark.parametrize("floor", [1e-6, 0.5])
def test_factor_oracle_equals_full_matrix_oracle_including_floor_and_empty(floor):
    torch.manual_seed(41)
    p = torch.randn(7, 4) * 0.1 + torch.tensor([0.3, 1.5, 120.37, 139.37])
    p[0, 2:] = -100
    p[1, 2:] = torch.tensor([-0.999, -0.999])
    layer = scenarios.model(p, "strip", n=257, out=259, floor=floor)
    x, dy = torch.randn(3, 257), torch.randn(3, 259)
    state = copy.deepcopy(layer.kernel).double()
    chart = layer.declaration().charts[0]
    a = fixture.oracle_vjp(state, p, x, dy, chart, chunk=3)
    b = fixture.square_oracle_vjp(state, p, x, dy, chart, chunk=3)
    for actual, truth in zip(b, a):
        torch.testing.assert_close(actual, truth, rtol=1e-10, atol=1e-11)


@scenarios.GPU
@pytest.mark.parametrize(
    "alias", ["prepared", "split1", "split4", "split8", "split16", "torch-g8-p8"]
)
@pytest.mark.parametrize("n", [1024, 2048])
def test_square_all_sites_y_dx_dp(alias, n):
    run = case(n)
    c = replace(run.case, atoms=19)
    p = fixture.initialize(c).cuda()
    op = fixture.fixture_operator(c)
    layer = CSTLinear(chart=op.charts[0], atoms=p, kernel=op.kernel, device="cuda")
    x = torch.randn(7, n, device="cuda", requires_grad=True)
    dy = torch.randn(7, n, device="cuda")
    truth = fixture.square_oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )
    y = Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=run.entry(alias).plan
    )
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in zip((y, dx, dp), truth):
        torch.testing.assert_close(actual.double(), expected, rtol=4e-4, atol=2e-5)


@pytest.fixture(params=["prepared", "split1", "split8", "torch-g8-p8"])
def square_route(request, monkeypatch):
    original = scenarios.model
    global_run = load_run(
        "benchmarks/cuda/linear/cases/profile-product-large-global-2048-rho3-split-k.json",
        "benchmarks/cuda/linear/plans-profile-product-large-global-split-k.json",
    )

    def model(p, family, **kwargs):
        kwargs.setdefault("n", 1041)
        kwargs.setdefault("out", 1025)
        return original(p, family, **kwargs)

    def execute(layer, x, family, **kwargs):
        selected = global_run if family == "global" else case()
        return Dispatcher(registry=REGISTRY).run(
            layer, LinearInputs(x), plan=selected.entry(request.param).plan
        )

    monkeypatch.setattr(scenarios, "model", model)
    monkeypatch.setattr(scenarios, "run", execute)
    monkeypatch.setattr(scenarios, "strip_oracle", fixture.square_oracle_vjp)


@pytest.mark.usefixtures("square_route")
class TestSquareSavedState:
    test_updates = staticmethod(
        scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths
    )
    test_snapshots = staticmethod(
        scenarios.test_old_forward_snapshots_keep_pitch_amplitude_and_parameters
    )
    test_live_pitch = staticmethod(
        scenarios.test_captured_strip_reads_changed_pitch_and_spacing_fallback
    )


def test_adapter_and_dispatch_decode_square_shape_and_reject_old_revision():
    import hashlib
    import json

    from benchmark_database_fixtures import artifact

    from benchmarks.cuda.linear.protocol import (
        LOCAL_OPTIMIZER_POLICY,
        SQUARE_STRIP_PRODUCT_ORACLE_SCOPE,
    )
    from benchmarks.database.adapters.linear import project
    from benchmarks.dispatch.generate import _context

    run = case()
    run = replace(run, case=replace(run.case, rounds=3), plans=(run.entry("split1"),))
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
            polar_update="fused",
            seed=run.case.seed,
            plan_id=None if is_dense else "split1",
            plan=None if is_dense else REGISTRY.dump_plan(run.entry("split1").plan),
        )
        if row is correct:
            row["result"].update(
                scope=SQUARE_STRIP_PRODUCT_ORACLE_SCOPE,
                polar_update={"max": 1e-6, "rel_l2": 1e-7},
            )
        else:
            row["result"].update(
                operator=op,
                rows=run.case.rows,
                atoms=None if is_dense else run.case.atoms,
                profile=None if is_dense else run.case.profile,
                reference="dense_linear" if is_dense else "split1",
                optimizer=asdict(run.case.optimizer),
                optimizer_policy="ordinary AdamW"
                if is_dense
                else LOCAL_OPTIMIZER_POLICY,
            )
    projection = project(data)
    assert projection.revision == projection.protocol["revision"] == 7
    row = {
        "case_declaration": projection.case,
        "environment": projection.records[1].environment,
        "adapter_revision": 7,
        "payload": projection.records[1].payload,
        "protocol": projection.protocol,
    }
    ctx = _context(row, "cuda_graph")
    assert ctx.operator.charts[0].shape == (1024, 1024)
    row["adapter_revision"] = 5
    with pytest.raises(ValueError, match="revision differs"):
        _context(row, "cuda_graph")
