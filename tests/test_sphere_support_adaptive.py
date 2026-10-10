"""Whole-chart oracles for exact support-count adaptive Sphere execution."""

import math
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import test_sphere_direct as direct
import test_sphere_direct_recompute as recompute
import torch

from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_algorithm import (
    SphereSupportAdaptiveAlgorithm,
    SphereSupportAdaptiveRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import context_from_tensors

cuda = direct.cuda


@pytest.fixture(autouse=True)
def adaptive_binding(monkeypatch):
    monkeypatch.setattr(direct, "SphereDirectAlgorithm", SphereSupportAdaptiveAlgorithm)
    monkeypatch.setattr(
        direct,
        "recipe",
        lambda options=(4, True), **kw: SphereSupportAdaptiveRecipe(**kw),
    )
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def test_adaptive_recipe_metadata_and_roundtrip():
    algorithm, registry = SphereSupportAdaptiveAlgorithm(), Registry()
    registry.register(algorithm)
    recipe = SphereSupportAdaptiveRecipe()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for field in fields(recipe):
        wrong = False if field.name == "merge_output_vjp" else 0
        with pytest.raises(ValueError):
            SphereSupportAdaptiveRecipe(**{field.name: wrong})
        with pytest.raises(ValueError):
            SphereSupportAdaptiveRecipe(
                **{field.name: 1 if field.name == "merge_output_vjp" else True}
            )
    with pytest.raises(TypeError):
        algorithm.validate_recipe(object())
    model, x, *_ = direct.fixture(17, 3.0, atoms=9)
    context = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    assert algorithm.supports(context, recipe).supported
    for bad in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, deterministic=True),
        replace(context, parameter_dim=5),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
    ):
        assert not algorithm.supports(bad, recipe).supported
    chart = context.operator.charts[0]
    too_large = replace(
        chart, shape=(32769,), coordinates=(chart.coordinates[0],) * 32769
    )
    wrong = replace(
        context,
        input_shape=(context.input_shape[0], 32769),
        input_strides=(32769, 1),
        operator=replace(
            context.operator,
            layout=replace(context.operator.layout, input_chart=too_large),
        ),
    )
    assert not algorithm.supports(wrong, recipe).supported
    assert algorithm.workspace_bound(None, recipe) is None


def test_adaptive_metadata_import_does_not_load_execution():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_algorithm import SphereSupportAdaptiveAlgorithm, SphereSupportAdaptiveRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
r = Registry(); a = SphereSupportAdaptiveAlgorithm(); r.register(a)
p = ExecutionPlan(a.id, a.revision, SphereSupportAdaptiveRecipe())
assert r.loads_plan(r.dumps_plan(p)) == p
assert 'triton' not in sys.modules
assert not any(n.endswith(('.adaptive_executor', '.adaptive_kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@cuda
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("size,sigma", [(17, 1.0), (128, 16.0)])
def test_adaptive_all_six_requested_gradients_and_broad_fallback(
    size, sigma, need_x, need_p
):
    direct.test_full_fp64_all_six_and_requested_gradients(
        (4, True), size, sigma, 9, need_x, need_p
    )


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 3, 4, 5])
def test_adaptive_empty_and_group_tails(atoms):
    direct.test_empty_atoms_and_group_tails(atoms)


@cuda
@pytest.mark.parametrize("gap", [0.0, 2**-9, 2**-8, 1.0])
def test_adaptive_singleton_subfloor_near_floor_and_empty(gap):
    direct.test_singleton_empty_and_both_norm_floor_branches((4, True), gap)


@cuda
def test_adaptive_rectangular_strides_widths_and_antipodes():
    direct.test_rectangular_strides_asymmetric_widths_and_antipodes((4, True))


@cuda
def test_adaptive_retained_forward_and_live_sites_radius_scalars():
    direct.test_retained_forward_owns_live_geometry_and_scalars((4, True))


@cuda
def test_adaptive_graph_reads_live_width_geometry_and_all_gradients():
    direct.test_graph_replays_live_widths_geometry_and_full_gradients((4, True))


@cuda
def test_adaptive_twenty_public_updates_and_moments():
    direct.test_twenty_public_updates_keep_parameters_moments_and_exact_clock((4, True))


@cuda
def test_adaptive_twenty_complete_graph_updates_match_public_reference():
    recompute.test_recomputed_phi_twenty_complete_graph_updates_match_public_reference()


@cuda
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_adaptive_mixed_input_output_full_overflow_signed_and_zero_amplitude(
    need_x, need_p
):
    direct.test_one_group_mixes_all_complete_overflow_paths_and_signed_amplitudes(
        (4, True), need_x, need_p
    )


def physical_model(
    input_sites, output_sites=None, *, radius=16.0, sigma=1.0, centers=None
):
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    charts = [
        chart_presets.points(
            sites,
            geometry=geometry_presets.sphere(
                2, radius=radius, representation="intrinsic"
            ),
        )
        for sites in (
            input_sites,
            input_sites if output_sites is None else output_sites,
        )
    ]
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        w_c=1e6,
        input_bounds=BandwidthBounds(
            minimum=sigma, birth=sigma, maximum=sigma, upper_floor=sigma
        ),
        profile=presets.profile(TriweightSpec()),
    )
    if centers is None:
        centers = [[0.0, 0.0, 0.0, 0.0]]
    p = torch.tensor([[0.3, 0.8, *center] for center in centers])
    return CSTLinear(*charts, atoms=p, kernel=kernel, backend="factored").cuda()


def checked_snapshots(model, x):
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_executor import (
        forward_snapshots,
    )

    evidence = {}
    y, _, _ = forward_snapshots(
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        SphereSupportAdaptiveRecipe(),
        diagnostics=evidence,
    )
    for sites, q, _, precision, index, phi, norm, count in evidence["sides"]:
        gap, raw = recompute.full_axis_raw(sites, q, precision)
        expected_count = (gap > 0).sum(1)
        torch.testing.assert_close(count.long(), expected_count, atol=0, rtol=0)
        torch.testing.assert_close(
            norm, raw.square().sum(1).sqrt(), atol=2e-7, rtol=4e-6
        )
        assert phi.numel() == 0 and phi.untyped_storage().nbytes() == 0
        for atom, n in enumerate(expected_count.tolist()):
            if n <= 64:
                actual_ids = index[atom, :n].int().sort().values
                truth_ids = torch.nonzero(gap[atom] > 0).flatten().int()
                torch.testing.assert_close(actual_ids, truth_ids, atol=0, rtol=0)
    dy = torch.linspace(-0.7, 0.9, y.numel(), device="cuda").reshape_as(y)
    actual = direct.binding(model)(x)
    direct.gate(
        (actual, *torch.autograd.grad(actual, (x, model.atoms.p), dy)),
        direct.oracle_vjp(model, x, dy),
    )
    direct.gate((y,), (direct.oracle_vjp(model, x, dy)[0],))
    return evidence


@cuda
@pytest.mark.parametrize(
    "positive,total,reason", [(1, 128, 0), (1, 129, 8), (16, 128, 0), (17, 128, 16)]
)
def test_adaptive_candidates_128_129_and_support_16_17_are_distinct(
    positive, total, reason
):
    # All sites share a queried bin. Negative corners are inside the AABB but
    # outside the spherical support ball; whole-chart oracle must exclude them.
    corner = torch.tensor([16.0, 0.8, 0.8], dtype=torch.float64)
    corner = (corner / corner.norm() * 16).tolist()
    sites = [[16.0, 0.0, 0.0]] * positive + [corner] * (total - positive)
    model = physical_model(sites)
    x = (
        torch.linspace(-0.3, 0.7, 3 * total, device="cuda")
        .reshape(3, total)
        .requires_grad_()
    )
    evidence = checked_snapshots(model, x)
    assert evidence["reason"].tolist() == [[reason, reason]]
    assert evidence["fallback"].tolist() == [bool(reason)]
    assert all(side[7].tolist() == [positive] for side in evidence["sides"])


@cuda
@pytest.mark.parametrize("sigma,reason", [(0.12, 0), (0.16, 4)])
def test_adaptive_physical_query_four_to_six_rows(sigma, reason):
    site = [0.8, 0.6, 0.0]
    angle = math.acos(0.8)
    model = physical_model(
        [site], radius=1.0, sigma=sigma, centers=[[angle, 0.0, angle, 0.0]]
    )
    x = torch.tensor([[0.7], [-0.2]], device="cuda", requires_grad=True)
    evidence = checked_snapshots(model, x)
    assert evidence["reason"].tolist() == [[reason, reason]]
    assert evidence["fallback"].tolist() == [bool(reason)]


@cuda
def test_adaptive_tiny_zero_one_two_four_counts_with_nontrivial_center_vjp():
    # Four isolated patches produce 16 input/output support-count combinations.
    r = 16.0
    angles = [0.0, 0.6, 1.2, 1.8]
    sites, counts = [], [0, 1, 2, 4]
    for theta, n in zip(angles, counts):
        for k in range(n):
            offset = 0.006 + k * 0.009
            sites.append(
                [r * math.cos(theta + offset), r * math.sin(theta + offset), 0.0]
            )
    centers = [[r * a, 0.0, r * b, 0.0] for a in angles for b in angles]
    model = physical_model(sites, centers=centers)
    with torch.no_grad():
        model.atoms.p[0, 0] = 0.0  # regular zero amplitude, nonzero radial coordinate
        model.atoms.p[1, 0] = -0.3
    x = (
        torch.linspace(-0.5, 0.9, 3 * len(sites), device="cuda")
        .reshape(3, -1)
        .requires_grad_()
    )
    evidence = checked_snapshots(model, x)
    assert evidence["sides"][0][7].tolist() == [n for n in counts for _ in counts]
    assert evidence["sides"][1][7].tolist() == counts * 4
    assert not evidence["fallback"].any()
    assert not evidence["reason"].any()
    dy = torch.linspace(-0.7, 0.9, 3 * len(sites), device="cuda").reshape(3, -1)
    truth = direct.oracle_vjp(model, x, dy)
    assert truth[2][:, 2:].abs().max() > 1e-4


def launch_raw_sides(sites, q, precision, radius, *, floor=1e-6):
    """Low-level arbitrary embedded points, with complete fallback executed.

    Deliberately permits centres outside the sphere to isolate exact query-row
    guards that a rectangular range cannot always reach in a physical chart.
    Raw/norm/output truth still uses every supplied point independently.
    """
    import triton

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_kernels import (
        build_index,
        flagged_forward,
        flagged_pack,
        tiny_forward,
    )

    n = len(sites)
    x = torch.linspace(-0.3, 0.7, n, device="cuda").reshape(1, n)
    sides = []
    for _ in range(2):
        index = torch.full((1, 64), -1, dtype=torch.int16, device="cuda")
        norm = torch.full((1,), -1.0, device="cuda")
        count = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        csr = build_index(sites, torch.tensor(radius, device="cuda"))
        sides.append((sites, q, precision, index, norm, count, *csr))
    h = torch.full((1, 1), -999.0, device="cuda")
    y = torch.zeros_like(x)
    fallback = torch.empty(1, dtype=torch.bool, device="cuda")
    reasons = torch.empty(1, 2, dtype=torch.uint8, device="cuda")
    tiny_forward[(1,)](
        x,
        *sides[0],
        *sides[1],
        torch.ones(1, device="cuda"),
        h,
        y,
        fallback,
        reasons,
        1,
        n,
        n,
        64,
        1,
        16,
        floor,
        floor,
        True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    tiny_y, tiny_h = y.clone(), h.clone()
    for ss, qq, pp, index, norm, count, *_ in sides:
        flagged_pack[(1,)](
            ss,
            qq,
            pp,
            index,
            norm,
            count,
            fallback,
            n,
            64,
            triton.next_power_of_2(n),
            num_warps=4,
            enable_fp_fusion=False,
        )
        gap, raw = recompute.full_axis_raw(ss, qq, pp)
        assert count.tolist() == (gap > 0).sum(1).tolist()
        ids = torch.nonzero(gap[0] > 0).flatten()
        torch.testing.assert_close(
            index[0, : len(ids)].long().sort().values, ids, atol=0, rtol=0
        )
        torch.testing.assert_close(
            norm, raw.square().sum(1).sqrt(), atol=2e-7, rtol=4e-6
        )
    empty_phi = torch.empty(0, device="cuda")
    contraction_sides = [
        (side[0], side[1], side[2], side[3], empty_phi, side[4], side[5])
        for side in sides
    ]
    flagged_forward[(1,)](
        x,
        *contraction_sides[0],
        *contraction_sides[1],
        torch.ones(1, device="cuda"),
        h,
        y,
        fallback,
        1,
        n,
        n,
        64,
        1,
        16,
        1,
        4,
        floor,
        floor,
        True,
        True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # FP64 arbitrary-point algebra is independent of query/CSR/packed helpers.
    raw64 = (
        (1 - (sites.double() - q.double()).square().sum(1) * precision.double())
        .clamp_min(0)
        .pow(3)
    )
    phi64 = raw64 / raw64.norm().clamp_min(floor)
    truth_y = (x.double() @ phi64[:, None]) * phi64[None, :]
    direct.gate((y,), (truth_y,))
    if bool(fallback[0]):
        assert not tiny_y.any() and float(tiny_h[0, 0]) == -999.0
    return sides, fallback, reasons


@cuda
@pytest.mark.parametrize("five_rows", [False, True])
def test_adaptive_exact_four_five_query_row_guard_on_arbitrary_points(five_rows):
    # Radius4 defines width1 bins. Clipped x has one row; y spans four or five.
    # Such a five-row span is not generally realizable on a physical sphere,
    # so the separate physical 4→6 test above covers actual chart execution.
    q = torch.tensor([[10.0, 0.5 if five_rows else 0.0, 0.0]], device="cuda")
    precision = torch.tensor([0.25 if five_rows else 1 / 1.4**2], device="cuda")
    _, fallback, reasons = launch_raw_sides(q.clone(), q, precision, 4.0)
    expected = 4 if five_rows else 0
    assert reasons.tolist() == [[expected, expected]]
    assert fallback.tolist() == [five_rows]


@cuda
def test_adaptive_preserves_positive_support_when_raw_square_flushes():
    sites = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda")
    q = torch.zeros(1, 3, device="cuda")
    precision = torch.nextafter(
        torch.ones(1, device="cuda"), torch.zeros(1, device="cuda")
    )
    # A wider cell grid avoids query overflow; norm is below, far from floor.
    sides, fallback, reasons = launch_raw_sides(sites, q, precision, 16.0)
    gap, raw = recompute.full_axis_raw(sites, q, precision)
    assert float(gap[0, 0]) == 2**-24 and float(raw[0, 0]) > 0
    assert not fallback.any() and not reasons.any()
    assert all(int(side[5][0]) == 1 and float(side[4][0]) == 0 for side in sides)


@cuda
def test_adaptive_fresh_csr_is_exact_site_bijection_with_empty_bins_and_tails():
    from collections import Counter

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_kernels import (
        build_index,
    )

    # Clipped outliers, boundaries, duplicate points and a non-block-sized tail.
    sites = torch.tensor(
        [[0, 0, 0], [-16, -16, -16], [16, 16, 16], [20, -20, 0], [-12, 0, 4]] * 25
        + [[1, 2, 3]],
        dtype=torch.float32,
        device="cuda",
    )
    for moved in (sites, sites.roll(17, 0) * 0.9):
        ids, offsets, descriptor = build_index(moved, torch.tensor(16.0, device="cuda"))
        assert ids.dtype == offsets.dtype == torch.int32
        assert ids.numel() == len(sites) and offsets.shape == (513,)
        assert offsets[0] == 0 and offsets[-1] == len(sites)
        assert (offsets[1:] >= offsets[:-1]).all()
        assert (offsets[1:] == offsets[:-1]).any()
        bins = torch.floor(((moved + 16) / 4).clamp(0, 7)).long()
        keys = (bins[:, 0] * 8 + bins[:, 1]) * 8 + bins[:, 2]
        assert sorted(ids.tolist()) == list(range(len(sites)))
        for k in range(512):
            actual = ids[int(offsets[k]) : int(offsets[k + 1])].tolist()
            truth = torch.nonzero(keys == k).flatten().tolist()
            assert Counter(actual) == Counter(truth)
        assert descriptor.tolist()[:2] == [-16.0, 4.0]


@cuda
def test_adaptive_never_allocates_phi_weight_or_weight_gradient():
    from torch.utils._python_dispatch import TorchDispatchMode

    allocations = []

    class NoMaterialization(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))

            def inspect(value):
                if isinstance(value, torch.Tensor):
                    allocations.append((tuple(value.shape), value.dtype))
                    assert tuple(value.shape) != (17, 17), "W/dW allocation"
                    assert not (
                        value.is_floating_point() and value.shape == (19, 64)
                    ), "Phi allocation"
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect(item)

            inspect(result)
            return result

    model, x, _, dy = direct.fixture(17, 1.0, batch=3, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    call = direct.binding(model)
    y = call(x)
    torch.autograd.grad(y, (x, model.atoms.p), dy)
    with NoMaterialization():
        y = call(x)
        actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    assert ((19, 64), torch.int16) in allocations
    assert ((19, 3), torch.float32) in allocations
    direct.gate(actual, direct.oracle_vjp(model, x, dy))


@cuda
def test_adaptive_near_floor_forces_complete_pack_without_changing_normalization():
    sites = torch.tensor([[math.sqrt(0.99), 0.0, 0.0]], device="cuda")
    q = torch.zeros(1, 3, device="cuda")
    _, fallback, reasons = launch_raw_sides(
        sites, q, torch.ones(1, device="cuda"), 16.0
    )
    assert fallback.tolist() == [True]
    assert reasons.tolist() == [[32, 32]]


@cuda
def test_adaptive_one_group_and_tail_mix_tiny_and_each_side_fallback():
    r = 16.0
    sites = [[r, 0.0, 0.0]] + [[0.0, r, 0.0]] * 17
    far = r * math.pi / 2
    choices = [(0, 0), (0, far), (far, 0), (far, far)] * 2 + [(0, 0)]
    centers = [[a, 0, b, 0] for a, b in choices]
    model = physical_model(sites, centers=centers)
    with torch.no_grad():
        model.atoms.p[0, 0] = 0
        model.atoms.p[1, 0] = -0.3
    x = (
        torch.linspace(-0.5, 0.9, 3 * len(sites), device="cuda")
        .reshape(3, -1)
        .requires_grad_()
    )
    evidence = checked_snapshots(model, x)
    assert evidence["reason"].tolist() == [[0, 0], [0, 16], [16, 0], [16, 16]] * 2 + [
        [0, 0]
    ]
    assert evidence["fallback"].tolist() == [False, True, True, True] * 2 + [False]
    assert evidence["sides"][0][7].tolist() == [1, 1, 17, 17] * 2 + [1]
    assert evidence["sides"][1][7].tolist() == [1, 17, 1, 17] * 2 + [1]


@cuda
def test_l4_compiled_adaptive_resources_and_fp32_cuda_core_instructions(
    monkeypatch, tmp_path
):
    import hashlib
    import json
    from pathlib import Path

    from torchcst._backends.cuda.algorithms.linear.sphere_polar import (
        adaptive_kernels,
        direct_kernels,
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

    names = (
        "describe",
        "histogram",
        "prefix",
        "scatter",
        "tiny_forward",
        "flagged_pack",
        "flagged_forward",
    )
    for name in names:
        monkeypatch.setattr(
            adaptive_kernels,
            name,
            CaptureCompiled(getattr(adaptive_kernels, name), name),
        )
    monkeypatch.setattr(
        direct_kernels, "backward", CaptureCompiled(direct_kernels.backward, "backward")
    )
    model, x, _, dy = direct.fixture(2048, 3.0, batch=32, atoms=int(0.05 * 2048**2))
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    y = direct.binding(model)(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    assert all(torch.isfinite(v).all() for v in (y, dx, dp))
    torch.cuda.synchronize()
    folder = Path(os.environ.get("CST_JOB_OUTPUT", tmp_path)) / "compiler-adaptive"
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, compiled in captured.items():
        ptx = compiled.asm["ptx"]
        file = folder / f"adaptive-{name}.ptx"
        file.write_text(ptx)
        records[name] = {
            "n_regs": compiled.n_regs,
            "n_spills": compiled.n_spills,
            "shared_bytes": compiled.metadata.shared,
            "ptx_file": file.name,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            **direct.ptx_arithmetic_report(ptx),
        }
    report = {
        "scope": "separate compiler diagnostic, not a time or adoption claim",
        "algorithm_id": SphereSupportAdaptiveAlgorithm().id,
        "recipe": vars(SphereSupportAdaptiveRecipe()),
        "gpu": torch.cuda.get_device_name(0),
        "size": 2048,
        "batch": 32,
        "atoms": int(0.05 * 2048**2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernels": records,
    }
    (folder / "compiler-adaptive.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    assert set(records) == set(names) | {"backward"}
    for name, record in records.items():
        assert (
            record["n_regs"] > 0
            and record["n_spills"] >= 0
            and record["shared_bytes"] >= 0
        )
        assert not record["mma_present"] and not record["tf32_present"]
        if name in {"tiny_forward", "flagged_pack", "flagged_forward", "backward"}:
            assert all(record["fp32_instruction_samples"].values())


@cuda
@pytest.mark.parametrize(
    "radius,precision,reason", [(0.0, 1.0, 1), (16.0, 0.0, 2), (16.0, 1e30, 2)]
)
def test_adaptive_invalid_descriptor_and_unsafe_precision_fall_back_exactly(
    radius, precision, reason
):
    # Zero distance makes the complete raw profile finite even for the unsafe
    # precisions; this isolates dispatch safety from intentionally NaN algebra.
    sites = torch.zeros(1, 3, device="cuda")
    _, fallback, reasons = launch_raw_sides(
        sites, sites.clone(), torch.tensor([precision], device="cuda"), radius
    )
    assert fallback.tolist() == [True]
    assert reasons.tolist() == [[reason, reason]]
