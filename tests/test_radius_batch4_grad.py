"""The explicit batched schedule shares derivative geometry, not responses."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_batched_cached import bounded_atoms
from test_radius_cell_mass_cuda import GPU, axis_plan, reference

from benchmarks.training.cell_mass_mnist import make_radius_cst
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
)


# N128/M8 start=0 retains 510*17 doubles (>48 KiB). Start=1 still
# exercises compact low-level banks while fitting the real native budgets.
POLAR_STARTS = (1, 2, 8)


@pytest.mark.parametrize("start", POLAR_STARTS)
def test_polar_fixture_fits_projection_and_both_expansion_budgets(start):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.expansion_layout import (
        batched_coefficient_layout,
    )

    layer = make_radius_cst(128, 7)
    plan = hierarchy_from_operator(layer.declaration())
    for axis in (plan.input, plan.output):
        elements = axis.active_nodes(start) * axis.basis_count
        # Saved moments plus responses share projection scratch. Cached2 also
        # holds atom profiles and one transposed coefficient warp; schedule3
        # has its own metadata-only native-layout mirror.
        assert 8 * (7 + elements) <= 48 * 1024
        cached_bytes = 8 * elements + 32 * ((axis.modes + 3) * 8 + 4)
        if elements:
            cached_bytes += axis.basis_count * 33 * 8
        assert cached_bytes <= 48 * 1024
        assert batched_coefficient_layout(axis, start, 32).bytes <= 48 * 1024


def assert_probe_agreement(actual, reference):
    """Keep FP32 bytes exact and allow a small FP64 roundoff-scale difference.

    gamma_32 = 32*u/(1-32*u), u=eps/2, gives a 16-epsilon output-scale
    diagnostic budget. The unit floor handles near-zero/cancelled components.
    This is an implementation-comparison allowance, not a forward-error proof
    or evidence of a particular FMA transformation. Independent VJP tolerances
    below are unchanged and remain a separate gate.
    """
    assert actual.dtype == reference.dtype and actual.shape == reference.shape
    assert bool(torch.isfinite(actual).all()) and bool(
        torch.isfinite(reference).all()
    ), "non-finite gradient comparison"
    if actual.dtype == torch.float32:
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        ), "FP32 probe must agree byte-for-byte"
    else:
        assert actual.dtype == torch.float64
        unit_roundoff = torch.finfo(torch.float64).eps / 2
        gamma = 32 * unit_roundoff / (1 - 32 * unit_roundoff)
        scale = reference.abs().clamp_min(1)
        assert bool(((actual - reference).abs() <= gamma * scale).all()), (
            "FP64 probe exceeds gamma_32 roundoff budget"
        )


def test_probe_accepts_fp64_roundoff_but_keeps_fp32_bits_exact():
    reference = torch.tensor([0.0, 1.0, 32.0, -32.0], dtype=torch.float64)
    actual = reference.clone()
    actual[0] = 1e-15
    for _ in range(4):
        actual[1:] = torch.nextafter(
            actual[1:], torch.full_like(actual[1:], float("inf"))
        )
    assert_probe_agreement(actual, reference)
    reference32 = reference.float()
    assert_probe_agreement(reference32.clone(), reference32)
    drift32 = reference32.clone()
    drift32[1] = torch.nextafter(drift32[1], torch.full_like(drift32[1], float("inf")))
    with pytest.raises(AssertionError, match="byte-for-byte"):
        assert_probe_agreement(drift32, reference32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "reference", "both"])
def test_probe_rejects_nonfinite_values_even_when_both_sides_match(dtype, value, side):
    actual = torch.ones(2, dtype=dtype)
    reference = actual.clone()
    if side in ("actual", "both"):
        actual[0] = value
    if side in ("reference", "both"):
        reference[0] = value
    with pytest.raises(AssertionError, match="non-finite"):
        assert_probe_agreement(actual, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_probe_rejects_material_drift(dtype):
    reference = torch.tensor([0.0, 1.0, 32.0], dtype=dtype)
    actual = reference.clone()
    actual[0] = 1e-10
    with pytest.raises(AssertionError):
        assert_probe_agreement(actual, reference)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 4, 5, 32])
@pytest.mark.parametrize(
    "sites,modes,merge,start",
    [(128, 8, 2, 2), (128, 8, 2, 5), (128, 8, 2, 8), (16, 16, 2, 0), (27, 3, 3, 1)],
)
def test_complete_gradient_probe_matches_scalar_and_preserves_all_caches(
    dtype, batch, sites, modes, merge, start
):
    axis = axis_plan(sites, modes=modes, merge=merge, spacing=0.37, origin=-1.2)
    plan = HierarchyPlan(input=axis, output=axis)
    p = bounded_atoms(plan, dtype).repeat(5, 1).cuda()
    p[0, 0] = 0  # zero amplitude must retain its amplitude gradient
    gen = torch.Generator().manual_seed(517)
    x = torch.randn(batch, sites, dtype=dtype, generator=gen).cuda()
    dy = torch.randn(batch, sites, dtype=dtype, generator=gen).cuda()
    native = ex._extension()
    config = axis.native_config()
    zx, h = native.shared_project_parallel(x, p, config, False, start, True)
    zy, g = native.shared_project_parallel(dy, p, config, True, start, True)
    arguments = (p, x, dy, zx, zy, h, g, config, config, start, start)
    before = [value.clone() for value in arguments[:7]]
    old, actual = native.parameter_grad_comparison(*arguments)
    # Per-row derivative order and final b=0..B-1 order are intentionally kept.
    # FP64 compiler roundoff is checked separately from the independent VJP.
    assert_probe_agreement(actual, old)
    for value, saved in zip(arguments[:7], before):
        # Unconsumed moment slots may contain NaN payloads; immutability is a
        # storage-byte property and must not depend on NaN floating equality.
        assert torch.equal(value.view(torch.uint8), saved.view(torch.uint8))
    expected_p = p.detach().cpu().requires_grad_()
    expected_y = reference(plan, x.cpu(), expected_p)
    expected = torch.autograd.grad(expected_y, expected_p, dy.cpu().double())[0]
    tolerance = (
        dict(atol=3e-5, rtol=3e-4)
        if dtype == torch.float32
        else dict(atol=2e-9, rtol=2e-8)
    )
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)


@GPU
@pytest.mark.parametrize(
    "empty_atoms,empty_batch", [(True, False), (False, True), (True, True)]
)
def test_parameter_probe_handles_empty_work(empty_atoms, empty_batch):
    axis = axis_plan(16, modes=3)
    plan = HierarchyPlan(input=axis, output=axis)
    p = (
        torch.empty(0, 4, dtype=torch.float64)
        if empty_atoms
        else bounded_atoms(plan, torch.float64)
    )
    p = p.cuda()
    x = torch.randn(0 if empty_batch else 5, 16, dtype=p.dtype, device=p.device)
    native = ex._extension()
    config = axis.native_config()
    zx, h = native.shared_project_parallel(x, p, config, False, 2, True)
    zy, g = native.shared_project_parallel(x, p, config, True, 2, True)
    old, actual = native.parameter_grad_comparison(
        p, x, x, zx, zy, h, g, config, config, 2, 2
    )
    assert torch.equal(actual, old)
    assert torch.equal(actual, torch.zeros_like(p))


@GPU
@pytest.mark.parametrize(
    "native,checked", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("start", [2, 5])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_batch4_gradient_full_vjps_retained_and_offloaded_saved_tensors(
    native, checked, offload, start, needs_x, needs_p
):
    axis = axis_plan(128, modes=8, spacing=0.37, origin=-1.2)
    plan = HierarchyPlan(input=axis, output=axis)
    gen = torch.Generator().manual_seed(909)
    host_x = torch.randn(5, 128, dtype=torch.float64, generator=gen).requires_grad_(
        needs_x
    )
    host_p = bounded_atoms(plan, torch.float64).requires_grad_(needs_p)
    host_dy = torch.randn(5, 128, dtype=torch.float64, generator=gen)
    expected_y = reference(plan, host_x, host_p)
    expected_grads = torch.autograd.grad(
        expected_y, tuple(t for t in (host_x, host_p) if t.requires_grad), host_dy
    )
    x = host_x.detach().cuda().requires_grad_(needs_x)
    p = host_p.detach().cuda().requires_grad_(needs_p)
    selected = tuple(t for t in (x, p) if t.requires_grad)
    recipe = replace(
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
        native_autograd=native,
        native_entry=checked,
        input_direct_levels=start,
        output_direct_levels=start,
    )

    def pack(value):
        return (value.detach().cpu().clone(), value.device) if offload else value

    def unpack(value):
        return value[0].to(value[1]) if offload else value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual = ex.radius_linear(plan, x, p, recipe=recipe)
    dy = host_dy.cuda()
    gradients = torch.autograd.grad(actual, selected, dy, retain_graph=True)
    again = torch.autograd.grad(actual, selected, dy)
    for value, repeated in zip(gradients, again):
        torch.testing.assert_close(value, repeated, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu(), expected_y, atol=2e-9, rtol=2e-8)
    for value, expected in zip(gradients, expected_grads):
        torch.testing.assert_close(value.cpu(), expected, atol=2e-9, rtol=2e-8)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("start", POLAR_STARTS)
@pytest.mark.parametrize("batch", [1, 4, 5, 31, 32, 33])
def test_batch4_polar_full_vjp_and_probe_match_public_cpu(dtype, start, batch):
    layer = make_radius_cst(128, 7, dtype=dtype)
    with torch.no_grad():
        layer.atoms.p[:, :2] = torch.tensor(
            [[0, 1], [0, 2], [1, 0], [3, 4], [-1, 0], [0, 1.5], [0.3, 0.4]], dtype=dtype
        )
        layer.atoms.p[:, 2:] = torch.tensor(
            [
                [0, 128],
                [-128, -1e-7],
                [127.99, 128.01],
                [512, -512],
                [-0.0, 0],
                [256, -256],
                [7.3, 8.2],
            ],
            dtype=dtype,
        )
    plan = hierarchy_from_operator(layer.declaration())
    layer.selector = None
    layer.backend = "factored"
    gen = torch.Generator().manual_seed(275)
    host_x = torch.randn(batch, 128, dtype=dtype, generator=gen).requires_grad_()
    host_dy = torch.randn(batch, 128, dtype=dtype, generator=gen)
    expected_y = layer(host_x)
    expected_grads = torch.autograd.grad(expected_y, (host_x, layer.atoms.p), host_dy)
    x = host_x.detach().cuda().requires_grad_()
    p = layer.atoms.p.detach().cuda().requires_grad_()
    dy = host_dy.cuda()
    recipe = replace(
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
        native_autograd=True,
        native_entry=True,
        input_direct_levels=start,
        output_direct_levels=start,
    )
    actual = ex.radius_linear(plan, x, p, recipe=recipe)
    gradients = torch.autograd.grad(actual, (x, p), dy, retain_graph=True)
    repeated = torch.autograd.grad(actual, (x, p), dy)
    for first, second in zip(gradients, repeated):
        assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    # The task gradient is tangent to (s,t); radius querying remains live in
    # the separate public CPU contract test, not an additional task path.
    radial = (gradients[1][:, :2] * p.detach()[:, :2]).sum(1)
    torch.testing.assert_close(
        radial,
        torch.zeros_like(radial),
        atol=3e-5 if dtype == torch.float32 else 2e-12,
        rtol=0,
    )
    tolerance = (
        dict(atol=3e-5, rtol=3e-4)
        if dtype == torch.float32
        else dict(atol=2e-9, rtol=2e-8)
    )
    torch.testing.assert_close(actual.cpu(), expected_y, **tolerance)
    for value, expected in zip(gradients, expected_grads):
        torch.testing.assert_close(value.cpu(), expected, **tolerance)
    native = ex._extension()
    config_i, config_o = plan.input.native_config(), plan.output.native_config()
    zx, h = native.shared_project_parallel(
        x.detach(), p.detach(), config_i, False, start, True
    )
    zy, g = native.shared_project_parallel(dy, p.detach(), config_o, True, start, True)
    old, candidate = native.parameter_grad_comparison(
        p.detach(), x.detach(), dy, zx, zy, h, g, config_i, config_o, start, start
    )
    assert_probe_agreement(candidate, old)


@GPU
@pytest.mark.parametrize("start", POLAR_STARTS)
def test_polar_task_specialization_preserves_public_adamw_state(start):
    """Scalar and specialized task gradients feed identical public updates.

    This isolates the eliminated derivative work using the existing probe;
    independent full-VJP tests above gate the forward/backward mathematics.
    """
    from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )

    models = [make_radius_cst(128, 7).cuda() for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [
        CSTAdamW(
            model.parameters(),
            model=model,
            lr=1e-4,
            weight_decay=0.0,
            amplitude_weight_decay=0.01,
            fused=True,
            check_finite=False,
            update_selector=FixedSelector(
                RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
            ),
        )
        for model in models
    ]
    plan = hierarchy_from_operator(models[0].declaration())
    config_i, config_o = plan.input.native_config(), plan.output.native_config()
    native = ex._extension()
    gen = torch.Generator().manual_seed(458)
    for batch in (4, 5, 32):
        x = torch.randn(batch, 128, generator=gen).cuda()
        dy = torch.randn(batch, 128, generator=gen).cuda()
        p = models[0].atoms.p.detach()
        zx, h = native.shared_project_parallel(x, p, config_i, False, start, True)
        zy, g = native.shared_project_parallel(dy, p, config_o, True, start, True)
        gradients = native.parameter_grad_comparison(
            p, x, dy, zx, zy, h, g, config_i, config_o, start, start
        )
        assert_probe_agreement(gradients[1], gradients[0])
        for model, optimizer, gradient in zip(models, optimizers, gradients):
            model.atoms.p.grad = gradient
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert torch.equal(
            models[0].atoms.p.view(torch.uint8), models[1].atoms.p.view(torch.uint8)
        )
        states = [opt.state[model.atoms.p] for model, opt in zip(models, optimizers)]
        assert states[0].keys() == states[1].keys()
        for key in states[0]:
            assert torch.equal(
                states[0][key].reshape(-1).view(torch.uint8),
                states[1][key].reshape(-1).view(torch.uint8),
            ), key
