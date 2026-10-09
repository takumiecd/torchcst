"""Pure compact full-scan control: unchanged float kernels and full gradients."""

import copy
from dataclasses import fields, replace

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp
from torchcst._backends.cuda.algorithms.linear.sphere_polar.compact_algorithm import (
    SphereGroupedCompactAlgorithm,
    SphereGroupedCompactRecipe,
)
from torchcst._backends.cuda.algorithms.linear.sphere_polar.grouped_algorithm import (
    SphereGroupedWeightRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan
from torchcst.operators.context import context_from_tensors
from torchcst.operators.execution import LinearBinding, LinearInputs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def binding(model):
    registry = Registry()
    algorithm = SphereGroupedCompactAlgorithm()
    registry.register(algorithm)
    dispatcher = Dispatcher(registry=registry)
    live = LinearBinding(model.operator, model.atoms.p)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, SphereGroupedCompactRecipe())
    return lambda x: dispatcher.run(live, LinearInputs(x), plan=plan)


def gate(actual, expected):
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if not a.numel():
            continue
        e = error(a, b)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


def test_compact_fixed_recipe_preserves_original_recipe():
    recipe = SphereGroupedCompactRecipe()
    assert tuple(vars(recipe).values()) == (64, 16, 8, 16)
    algorithm = SphereGroupedCompactAlgorithm()
    algorithm.validate_recipe(recipe)
    assert algorithm.workspace_bound(None, recipe) is None
    with pytest.raises(TypeError):
        algorithm.validate_recipe(SphereGroupedWeightRecipe())
    for field in fields(recipe):
        for value in (
            True,
            float(getattr(recipe, field.name)),
            getattr(recipe, field.name) + 1,
        ):
            with pytest.raises(ValueError):
                SphereGroupedCompactRecipe(**{field.name: value})
    with pytest.raises(ValueError, match="G4/T16"):
        SphereGroupedWeightRecipe(atom_group=8, index_bits=16)


def test_compact_metadata_geometry_and_lossless_domain():
    model, x, *_ = fixture(17, 3.0, atoms=9)
    context = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    algorithm, recipe = SphereGroupedCompactAlgorithm(), SphereGroupedCompactRecipe()
    assert algorithm.supports(context, recipe).supported
    for bad in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, deterministic=True),
        replace(context, parameter_dim=5),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
        replace(context, precision=replace(context.precision, autocast=True)),
    ):
        assert not algorithm.supports(bad, recipe).supported
    for size in (32768, 32769):
        chart = replace(
            context.operator.charts[0],
            shape=(size,),
            coordinates=((1.0, 0.0, 0.0),) * size,
        )
        bad = replace(
            context,
            input_shape=(context.m, size),
            input_strides=(size, 1),
            operator=replace(
                context.operator,
                layout=replace(context.operator.layout, input_chart=chart),
            ),
        )
        result = algorithm.supports(bad, recipe)
        # Existing measured scope stays<=2048. The lossless storage bound is a
        # separate guard, not permission to silently expand that domain.
        assert not result.supported
        assert any("32768" in reason for reason in result.reasons) == (size > 32768)
    chart = replace(
        context.operator.charts[0],
        geometry=replace(context.operator.charts[0].geometry, revision=2),
    )
    bad = replace(
        context,
        operator=replace(
            context.operator, layout=replace(context.operator.layout, input_chart=chart)
        ),
    )
    assert not algorithm.supports(bad, recipe).supported


@cuda
@pytest.mark.parametrize(
    "size,sigma,atoms",
    [(17, 1.0, 19), (64, 3.0, 9), (128, 16.0, 17), (1024, 3.0, 9), (2048, 3.0, 17)],
)
@pytest.mark.parametrize("need_x,need_p", [(True, True), (True, False), (False, True)])
def test_full_fp64_values_and_requested_gradients(size, sigma, atoms, need_x, need_p):
    model, x, _, dy = fixture(size, sigma, batch=3, atoms=atoms)
    model = model.cuda()
    x, dy = x.cuda().T.contiguous().T.requires_grad_(need_x), dy.cuda().T.contiguous().T
    expected = oracle_vjp(model, x, dy)
    model.atoms.p.requires_grad_(need_p)
    y = binding(model)(x)
    targets = tuple(t for t, needed in ((x, need_x), (model.atoms.p, need_p)) if needed)
    grads = torch.autograd.grad(y, targets, dy)
    truth = tuple(
        t for t, needed in ((expected[1], need_x), (expected[2], need_p)) if needed
    )
    gate((y, *grads), (expected[0], *truth))


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 7, 8, 9])
def test_zero_atoms_and_group_tails(atoms):
    model, x, _, dy = fixture(17, 1.0, batch=2, atoms=max(atoms, 1))
    if atoms == 0:
        model.atoms.p = torch.nn.Parameter(model.atoms.p[:0].clone())
    assert model.atom_count == atoms
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    y = binding(model)(x)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    if atoms:
        gate(actual, oracle_vjp(model, x, dy))
    else:
        assert actual[2].shape == (0, 6)
        assert all(not torch.count_nonzero(value) for value in actual)


@cuda
def test_lossless_preparation_and_explicit_complete_overflow():
    from types import SimpleNamespace

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    for size, sigma in ((64, 3.0), (128, 16.0)):
        model, x, *_ = fixture(size, sigma, batch=2, atoms=19)
        model, x = model.cuda(), x.cuda()
        args = (
            x,
            model.atoms.p,
            model.kernel,
            model.cst_charts(),
            SphereGroupedCompactRecipe(),
        )
        compact, full = (
            prepare(*args, index_dtype=torch.int16),
            prepare(*args, index_dtype=torch.int32),
        )
        for one, two in zip(compact[5], full[5]):
            assert one[4].dtype == torch.int16 and two[4].dtype == torch.int32
            assert (
                one[4].numel() * one[4].element_size() * 2
                == two[4].numel() * two[4].element_size()
            )
            for i in (0, 1, 2, 3, 6, 7):
                assert torch.equal(one[i], two[i])
            for row, count in enumerate(one[7].tolist()):
                if count <= 64:
                    assert torch.equal(one[4][row, :count].int(), two[4][row, :count])
                    assert torch.equal(one[5][row, :count], two[5][row, :count])
            if size == 128:
                assert torch.equal(one[7], torch.full_like(one[7], 128))
    with pytest.raises(ValueError, match="32768"):
        prepare(
            x,
            model.atoms.p,
            model.kernel,
            [SimpleNamespace(coordinates=torch.empty(32769, 3))],
            SphereGroupedCompactRecipe(),
            index_dtype=torch.int16,
        )


@cuda
def test_regular_zero_amplitude_and_retained_snapshot():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    model, x, _, dy = fixture(64, 3.0, batch=4, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    with torch.no_grad():
        model.atoms.p[0, 0].zero_()
    assert model.atoms.p[0, 1] != 0
    assert (
        float(
            prepare(
                x,
                model.atoms.p,
                model.kernel,
                model.cst_charts(),
                SphereGroupedCompactRecipe(),
                index_dtype=torch.int16,
            )[2][0]
        )
        == 0.0
    )
    expected = oracle_vjp(model, x, dy)
    call = binding(model)
    y = call(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    gate((y, *first), expected)
    with torch.no_grad():
        model.atoms.p.add_(0.001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        model.kernel.scalar("sigma_max_input").mul_(0.9)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
            chart.geometry.radius.mul_(0.9)
    newer = call(x)
    gate(
        (newer, *torch.autograd.grad(newer, (x, model.atoms.p), dy)),
        oracle_vjp(model, x, dy),
    )
    after = torch.autograd.grad(y, (x, model.atoms.p), dy)
    for a, b in zip(first, after):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    gate((y, *after), expected)


@cuda
def test_graph_reads_live_centers_widths_sites_radii_and_scalars():
    model, x, _, _ = fixture(64, 3.0, batch=4, atoms=19)
    model, x = model.cuda(), x.cuda().requires_grad_()
    call = binding(model)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.grad, model.atoms.p.grad = None, None
            call(x).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    x.grad, model.atoms.p.grad = None, None
    with torch.cuda.graph(graph, stream=stream):
        y = call(x)
        y.sum().backward()
    for scale in (0.96, 1.07):
        with torch.no_grad():
            model.atoms.p[:, :2].mul_(scale)
            model.atoms.p[:, 2:].add_(0.01)
            model.kernel.scalar("amplitude_max").mul_(scale)
            model.kernel.scalar("sigma_max_output").mul_(scale)
            for chart in model.cst_charts():
                chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
                chart.geometry.radius.mul_(scale)
        graph.replay()
        torch.cuda.synchronize()
        gate((y, x.grad, model.atoms.p.grad), oracle_vjp(model, x, torch.ones_like(y)))


@cuda
def test_twenty_public_optimizer_updates():
    from torchcst import CSTOptimizer

    base, x, target, _ = fixture(32, 3.0, batch=4)
    models = [copy.deepcopy(base).cuda() for _ in range(2)]
    optimizers = [
        CSTOptimizer(torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=False), model=m)
        for m in models
    ]
    xs = [x.cuda().requires_grad_() for _ in models]
    target = target.cuda()
    call = binding(models[1])
    for _ in range(20):
        for model, xx, opt, fn in zip(models, xs, optimizers, (models[0], call)):
            opt.zero_grad(set_to_none=True)
            xx.grad = None
            (fn(xx) - target).square().mean().backward()
            opt.step()
        torch.testing.assert_close(
            models[0].atoms.p, models[1].atoms.p, rtol=0, atol=2e-6
        )
        torch.testing.assert_close(xs[0].grad, xs[1].grad, rtol=0, atol=4e-4)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                optimizers[0].state[models[0].atoms.p][key],
                optimizers[1].state[models[1].atoms.p][key],
                rtol=0,
                atol=4e-4,
            )


@cuda
def test_lossless_max_id32767_uses_original_grouped_block():
    import triton
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.grouped_kernels import (
        _block,
    )

    @triton.jit
    def probe(S, Q, P, I, F, C, Out):
        a = tl.arange(0, 8)
        live = a < 1
        counts = tl.load(C + a, live, 0)
        idx, _, phi, gap, _, _, _, _ = _block(
            S, Q, P, I, F, a, live, counts, 0, 32768, 64, 16
        )
        t = tl.arange(0, 16)
        tl.store(Out + a[:, None] * 16 + t[None, :], idx)
        tl.store(Out + 128 + a[:, None] * 16 + t[None, :], phi * gap)

    sites = torch.zeros(32768, 3, device="cuda")
    sites[-1, 0] = 2.0
    q = torch.tensor([[2.0, 0.0, 0.0]], device="cuda")
    p = torch.ones(1, device="cuda")
    index = torch.zeros(1, 64, device="cuda", dtype=torch.int16)
    index[0, 0] = 32767
    phi = torch.ones(1, 64, device="cuda")
    count = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.empty(256, device="cuda")
    probe[(1,)](
        sites, q, p, index, phi, count, out, num_warps=4, enable_fp_fusion=False
    )
    assert out[0] == 32767 and out[128] == 1.0
