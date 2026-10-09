"""Lossless compact IDs preserve original supports, floors, snapshots and updates."""

import copy
from dataclasses import replace

import pytest
import torch
import test_torus_sparse_weight as previous
import test_torus_profile_product_onchip as existing

from benchmarks.cuda.linear.torus_profile_product import _strict_check, oracle_vjp
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.torus_profile_product.compact_ids.algorithm import (
    CompactIdsRecipe,
    TorusCompactIdsAlgorithm,
    site_id_eligible,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def run(layer, x):
    registry = Registry()
    algorithm = TorusCompactIdsAlgorithm()
    registry.register(algorithm)
    return Dispatcher(registry=registry).run(
        layer,
        LinearInputs(x),
        plan=ExecutionPlan(algorithm.id, "v1", CompactIdsRecipe()),
    )


def test_fixed_recipe_metadata_roundtrip_and_storage_bound():
    algorithm = TorusCompactIdsAlgorithm()
    registry = Registry()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, "v1", CompactIdsRecipe())
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for kwargs in (
        {"index_bits": True},
        {"index_bits": 32},
        {"circle_capacity": 16},
        {"section_capacity": 64},
        {"patch_tile": 32},
    ):
        with pytest.raises(ValueError):
            CompactIdsRecipe(**kwargs)
    for shape, allowed in (
        ((32768, 2048), True),
        ((2048, 32768), True),
        ((32769, 2048), False),
        ((True, 2048), False),
        ((0, 1), False),
    ):
        assert site_id_eligible(shape) == allowed
    from benchmarks.cuda.linear.manifest import load_run

    for n in (1024, 2048):
        for sigma in (3, 8):
            declaration = load_run(
                f"benchmarks/cuda/linear/cases/torus-compact-ids-{n}-sigma{sigma}.json",
                "benchmarks/cuda/linear/plans-torus-compact-ids.json",
            )
            assert [entry.id for entry in declaration.plans] == [
                "sparse-w",
                "compact16",
            ]
    layer = existing.fixture().float()
    context = layer.build_context(LinearInputs(torch.zeros(3, layer.in_features)))
    cuda = replace(context, device=DeviceInfo("cuda", 0, "NVIDIA L4", (8, 9), 58))
    assert algorithm.supports(cuda, CompactIdsRecipe()).supported
    assert not algorithm.supports(context, CompactIdsRecipe()).supported
    assert not algorithm.supports(
        replace(cuda, precision=replace(cuda.precision, allow_tf32=True)),
        CompactIdsRecipe(),
    ).supported
    chart = cuda.operator.charts[0]
    changed = replace(
        chart,
        shape=(32768, chart.shape[1]),
        axes=(replace(chart.axes[0], shape=(32768,)), chart.axes[1]),
        geometry=replace(chart.geometry, major_radius=32768.0),
    )
    operator = replace(
        cuda.operator, layout=replace(cuda.operator.layout, chart=changed)
    )
    assert not algorithm.supports(
        replace(cuda, operator=operator), CompactIdsRecipe()
    ).supported


@CUDA
@pytest.mark.parametrize("floor", [1e-6, 0.5, 100.0])
@pytest.mark.parametrize("batch", [1, 3, 32, 64])
def test_physical_oracle_all_five_gradients_and_floor(floor, batch):
    layer = existing.fixture(device="cuda", floor=floor).float()
    x = torch.randn(batch, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(batch, layer.out_features, device="cuda")
    y0, x0, p0 = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    y = run(layer, x)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in ((y, y0), (dx, x0), (dp, p0)):
        _strict_check(a, b, tol=4e-4)


@CUDA
def test_retained_forward_independent_gradients_and_empty(monkeypatch):
    monkeypatch.setattr(existing, "run", run)
    existing.test_previous_forward_snapshots_live_mutated_operands()
    for grads in ((True, False), (False, True), (True, True)):
        for empty in (False, True):
            existing.test_independent_gradient_requirements_and_empty(grads, empty)


@CUDA
def test_twenty_graph_steps_live_width_geometry_parameters_moments(monkeypatch):
    monkeypatch.setattr(previous, "run", run)
    previous.test_twenty_graph_steps_widths_geometry_public_moments()


@CUDA
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("sigma", [3, 8])
def test_square_seam_gap_and_complete_overflow(n, sigma, monkeypatch):
    monkeypatch.setattr(previous, "run", lambda layer, x, recipe: run(layer, x))
    previous.test_periodic_circle_window_and_exact_overflow(n, sigma, 256)


@CUDA
@pytest.mark.parametrize("period_shift", [-1.1, -0.9, 0.9, 1.1])
def test_existing_circle_interval_spacing_boundary(period_shift, monkeypatch):
    monkeypatch.setattr(previous, "run", run)
    previous.test_period_mismatch_spacing_boundary_near_seam(period_shift)


@CUDA
def test_live_grid_and_minor_snapshot_retained_backward():
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.torus_profile_product import (
        fixture_operator,
        initialize,
    )
    from torchcst import CSTLinear

    case = load_run(
        "benchmarks/cuda/linear/cases/torus-sparse-weight-1024-sigma3.json",
        "benchmarks/cuda/linear/plans-torus-sparse-weight.json",
    ).case
    op = fixture_operator(case)
    layer = CSTLinear(
        chart=op.charts[0], kernel=op.kernel, atoms=initialize(case)[:13], device="cuda"
    )
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y0, dx0, dp0 = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    y = run(layer, x)
    with torch.no_grad():
        layer.chart.axes[1].start.add_(0.13)
        layer.chart.axes[1].spacing.mul_(torch.tensor([1.125, 0.875], device="cuda"))
        layer.chart.geometry.minor_radius.mul_(1.07)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in ((y, y0), (dx, dx0), (dp, dp0)):
        _strict_check(a, b, tol=4e-4)
    y1, dx1, dp1 = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    fresh = run(layer, x)
    fresh_dx, fresh_dp = torch.autograd.grad(fresh, (x, layer.atoms.p), dy)
    for a, b in ((fresh, y1), (fresh_dx, dx1), (fresh_dp, dp1)):
        _strict_check(a, b, tol=4e-4)


@CUDA
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("sigma", [3, 8])
def test_original_pack_exact_ids_raw_stats_for_both_storage_dtypes(n, sigma):
    import triton
    from benchmarks.cuda.linear.scaling_comparison import fixture
    from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
        _queries,
        _Scalars,
    )
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.sparse_weight.kernels import (
        pack,
    )

    model, *_ = fixture("torus", n, sigma, batch=3, atoms=128)
    model = model.cuda()
    source = model.atoms.p.detach().clone()
    scalars = _Scalars(model.kernel, source)
    amp, alpha = polar._amplitude_and_alpha(scalars, source[:, :2])
    width, _, _ = polar._sigma_bounds(scalars, amp, alpha, side="input")
    precision = width.reciprocal().square().detach()
    major, minor, circle, sites, _ = _queries(model.chart, source)
    for side, cap in ((0, 64), (1, 256)):
        buffers = []
        for dtype in (torch.int32, torch.int16):
            index = torch.empty((128, cap), device="cuda", dtype=dtype)
            raw, stats = source.new_empty((128, cap)), source.new_empty((128, 5))
            pack[(128,)](
                source,
                precision,
                scalars.scalar("amplitude_max"),
                major,
                minor,
                circle,
                sites,
                index,
                raw,
                stats,
                model.chart.axes[0].start,
                model.chart.axes[0].spacing,
                model.chart.tile_pitch,
                N=n,
                CAP=cap,
                V=triton.next_power_of_2(n),
                TILE=model.chart.tile_shape[0],
                SIDE=side,
                num_warps=4,
                enable_fp_fusion=False,
            )
            buffers.append((index.cpu(), raw.cpu(), stats.cpu()))
        (i0, r0, s0), (i1, r1, s1) = buffers
        assert torch.equal(s0, s1), "FP32 raw norms/counts/derivative moments unchanged"
        for atom in range(128):
            count = int(s0[atom, 0])
            if count <= cap:
                assert torch.equal(i0[atom, :count], i1[atom, :count].int())
                assert torch.equal(r0[atom, :count], r1[atom, :count])


@CUDA
def test_maximum_lossless_id_widens_before_site_stride_and_weight_address():
    import triton as tr
    import triton.language as tl
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.compact_ids.contractions import (
        block,
    )

    @tr.jit
    def witness(Index, Raw, Sites, Weight, Out):
        idx, valid, _ = block(Index, Raw, 0, 0, 1, 32768, 64, 16, False)
        scalar = tl.load(Sites + 3 * idx + 2, valid, 0)
        weight = tl.load(Weight + idx * 2 + 1, valid, 0)
        off = tl.arange(0, 16)
        tl.store(Out + off, scalar + weight)

    index = torch.zeros((1, 64), device="cuda", dtype=torch.int16)
    index[0, 0] = 32767
    raw = torch.zeros((1, 64), device="cuda")
    sites = torch.zeros(32768 * 3, device="cuda")
    weight = torch.zeros(32768 * 2, device="cuda")
    sites[3 * 32767 + 2], weight[2 * 32767 + 1] = 5.0, 7.0
    output = torch.empty(16, device="cuda")
    witness[(1,)](index, raw, sites, weight, output)
    assert output[0].item() == 12.0
    assert not output[1:].any()


@CUDA
def test_candidate_copied_diagnostics_clock_and_original_state(monkeypatch):
    import test_scaling_comparison_protocol as common
    from benchmarks.cuda.linear import scaling_comparison as protocol

    monkeypatch.setattr(
        protocol,
        "baseline_plan",
        lambda *args: ExecutionPlan(
            TorusCompactIdsAlgorithm().id, "v1", CompactIdsRecipe()
        ),
    )
    common.test_copied_cst_phase_refreshes_metadata_without_extra_update("torus")


@CUDA
@pytest.mark.parametrize("shape", [(128, 64), (32, 1024), (128, 1024)])
def test_count_witnessed_circle_section_and_both_overflow(shape):
    import triton
    from torchcst import BandwidthBounds, CSTLinear, NormalizationSpec, presets
    from torchcst import chart_presets as charts
    from torchcst import geometry_presets as geometries
    from torchcst import pattern_presets as patterns
    from torchcst.kernels.profiles import TriweightSpec
    from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
        _queries,
        _Scalars,
    )
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.sparse_weight.kernels import (
        pack,
    )

    no, ni = shape
    geometry = geometries.torus(
        3,
        major_radius=40.0,
        minor_radius=8.0,
        circle_axis=0,
        representation="intrinsic",
    )
    chart = charts.strip(
        shape,
        (16, ni),
        axes=(
            patterns.line(no, low=0.0, high=float(no - 1)),
            patterns.grid((8, ni // 8), low=(-0.5, -0.5), high=(0.5, 0.5)),
        ),
        axis=0,
        tile_pitch=16.0,
        geometry=geometry,
    )
    bounds = BandwidthBounds(
        minimum=100.0, birth=100.0, maximum=100.0, upper_floor=100.0
    )
    kernel = presets.polar_torus_profile_product(
        profiles=(TriweightSpec(), TriweightSpec()),
        amplitude_max=1.0,
        bounds=bounds,
        w_c=0.5,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=1e-6
        ),
    )
    layer = CSTLinear(
        chart=chart,
        kernel=kernel,
        atoms=torch.tensor([[0.3, 1.5, 10.0, 0.2, -0.1]]),
        device="cuda",
    )
    source = layer.atoms.p.detach().clone()
    major, minor, circle, sites, _ = _queries(layer.chart, source)
    counts = []
    for side, n, cap in ((0, no, 64), (1, ni, 256)):
        idx = torch.empty((1, cap), device="cuda", dtype=torch.int16)
        raw, stats = source.new_empty((1, cap)), source.new_empty((1, 5))
        pack[(1,)](
            source,
            torch.full((1,), 1e-4, device="cuda"),
            _Scalars(layer.kernel, source).scalar("amplitude_max"),
            major,
            minor,
            circle,
            sites,
            idx,
            raw,
            stats,
            layer.chart.axes[0].start,
            layer.chart.axes[0].spacing,
            layer.chart.tile_pitch,
            N=n,
            CAP=cap,
            V=triton.next_power_of_2(n),
            TILE=16,
            SIDE=side,
            num_warps=4,
            enable_fp_fusion=False,
        )
        counts.append(int(stats[0, 0]))
    assert counts == [no, ni], (
        "sigma100 covers all sites and proves the exact overflow branch"
    )
    assert (counts[0] > 64, counts[1] > 256) == ((no > 64), (ni > 256))
    x = torch.randn(3, ni, device="cuda", requires_grad=True)
    dy = torch.randn(3, no, device="cuda")
    y0, dx0, dp0 = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    y = run(layer, x)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in ((y, y0), (dx, dx0), (dp, dp0)):
        _strict_check(actual, expected, tol=4e-4)
