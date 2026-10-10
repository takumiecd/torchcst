"""Independent full-Cartesian contracts for bounded regular Product execution."""

import copy
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import torch

from benchmarks.cuda.linear.global_profile_product import oracle_factors, oracle_vjp
from benchmarks.cuda.linear.profile_product import oracle_atoms
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    Dispatcher,
    LinearInputs,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.algorithm import (
    MatrixFreeProductAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.recipe import (
    MatrixFreeProductRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@pytest.fixture(autouse=True)
def ieee():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def model(
    p, *, ni=65, no=33, width=None, floor=1e-6, spacing=1, origin=0, device="cpu"
):
    bounds = BandwidthBounds(
        minimum=0.25 if width is None else width,
        birth=1 if width is None else width,
        maximum=16 if width is None else width,
        upper_floor=1 if width is None else width,
    )
    axes = tuple(
        pattern_presets.line(n, low=origin, high=origin + (n - 1) * spacing)
        for n in (no, ni)
    )
    return CSTLinear(
        chart=chart_presets.product((no, ni), axes),
        atoms=p,
        kernel=presets.polar_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1,
            bounds=bounds,
            w_c=1e6,
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=floor
            ),
            dormant_expansion_rate=0.02,
        ),
        device=device,
    )


def parameters(atoms, *, ni=65, no=33, origin=0):
    gen = torch.Generator().manual_seed(41)
    p = torch.randn(atoms, 4, generator=gen) * 0.07 + torch.tensor([0.3, 1.5, 0, 0])
    p[:, 2] = torch.rand(atoms, generator=gen) * (no - 2) + origin + 0.37
    p[:, 3] = torch.rand(atoms, generator=gen) * (ni - 2) + origin + 0.37
    if atoms > 2:
        p[0, 0], p[1, 0] = -0.3, 0.0
    return p


def run(layer, x, **kwargs):
    registry = Registry()
    algorithm = MatrixFreeProductAlgorithm()
    registry.register(algorithm)
    return Dispatcher(registry=registry).run(
        layer,
        LinearInputs(x),
        plan=ExecutionPlan(
            algorithm.id, algorithm.revision, MatrixFreeProductRecipe(**kwargs)
        ),
    )


def oracle(layer, x, dy):
    return oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )


def gate(actual, expected):
    for value, truth in zip(actual, expected, strict=True):
        assert value.shape == truth.shape
        assert torch.isfinite(value).all() and torch.isfinite(truth).all()
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@pytest.mark.parametrize("preparation", ["support", "full"])
@pytest.mark.parametrize("chunk", [256, 8192, 65536, 262144])
def test_matrix_free_recipe_roundtrip_and_strict_fields(preparation, chunk):
    algorithm, registry = MatrixFreeProductAlgorithm(), Registry()
    registry.register(algorithm)
    recipe = MatrixFreeProductRecipe(preparation=preparation, atom_chunk=chunk)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for field in fields(recipe):
        bad = "approximate" if field.name == "preparation" else True
        with pytest.raises(ValueError):
            MatrixFreeProductRecipe(**{field.name: bad})
    for name in ("atom_chunk", "owner_atoms", "owner_splits", "owner_capacity"):
        with pytest.raises(ValueError):
            MatrixFreeProductRecipe(**{name: 0})
    with pytest.raises(TypeError):
        algorithm.validate_recipe(object())


def test_matrix_free_domain_and_precision_metadata():
    algorithm, recipe = MatrixFreeProductAlgorithm(), MatrixFreeProductRecipe()
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24]]), ni=8192, no=33)
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(3, 8192))),
        device=DeviceInfo("cuda", 0),
    )
    assert algorithm.supports(context, recipe).supported
    for wrong in (
        replace(context, device=DeviceInfo("cpu", None)),
        replace(context, dtype=torch.float64),
        replace(context, atom_count=4194305),
        replace(context, parameter_dim=6),
        replace(context, precision=replace(context.precision, allow_tf32=True)),
        replace(context, precision=replace(context.precision, autocast=True)),
    ):
        assert not algorithm.supports(wrong, recipe).supported
    for n in (8193,):
        other = model(layer.atoms.p.detach().clone(), ni=n)
        bad = replace(
            other.build_context(LinearInputs(torch.zeros(3, n))),
            device=DeviceInfo("cuda", 0),
        )
        assert not algorithm.supports(bad, recipe).supported


def test_matrix_free_declaration_import_does_not_load_gpu_execution():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.algorithm import MatrixFreeProductAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.recipe import MatrixFreeProductRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
a=MatrixFreeProductAlgorithm(); r=Registry(); r.register(a)
p=ExecutionPlan(a.id,a.revision,MatrixFreeProductRecipe())
assert r.loads_plan(r.dumps_plan(p))==p
assert 'triton' not in sys.modules
assert not any(n.endswith(('regular_product_matrix_free.executor','regular_product_matrix_free.kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@pytest.mark.parametrize("floor,width", [(1e-6, 4), (0.5, 1), (1e-6, 400)])
def test_independent_oracle_uses_one_full_cartesian_norm_floor(floor, width):
    layer = model(
        torch.tensor([[0.3, 1.5, -0.39, -0.39], [-0.4, 1.4, 2.3, 5.7]]),
        ni=7,
        no=3,
        width=width,
        floor=floor,
    )
    p = layer.atoms.p.detach().double().requires_grad_()
    value = copy.deepcopy(layer.kernel).double()
    u, v, scale = oracle_factors(value, p, layer.declaration().charts[0])
    factored = scale[:, None, None] * u[:, :, None] * v[:, None, :]
    enumerated = oracle_atoms(
        value,
        p,
        1,
        output_sites=torch.arange(3, dtype=torch.float64),
        input_sites=torch.arange(7, dtype=torch.float64),
    )
    torch.testing.assert_close(factored, enumerated, atol=1e-12, rtol=1e-12)
    cotangent = torch.linspace(
        -0.7, 0.9, factored.numel(), dtype=torch.float64
    ).reshape_as(factored)
    (first,) = torch.autograd.grad(factored, p, cotangent, retain_graph=True)
    (second,) = torch.autograd.grad(enumerated, p, cotangent)
    torch.testing.assert_close(first, second, atol=1e-11, rtol=1e-11)


@GPU
@pytest.mark.parametrize("preparation", ["support", "full"])
@pytest.mark.parametrize("atoms", [255, 256, 257, 513])
@pytest.mark.parametrize("atom_group", [4, 8])
def test_matrix_free_chunk_tails_full_values_strides_and_all_four_gradients(
    preparation, atoms, atom_group
):
    layer = model(parameters(atoms), device="cuda")
    gen = torch.Generator().manual_seed(43)
    x = torch.randn(7, 130, generator=gen).cuda()[:, ::2].detach().requires_grad_()
    dy = torch.randn(33, 7, generator=gen).cuda().T
    expected = oracle(layer, x, dy)
    y = run(layer, x, preparation=preparation, atom_chunk=256, atom_group=atom_group)
    gate((y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)), expected)
    assert expected[2][1, 0].abs() > 1e-8


@GPU
@pytest.mark.parametrize("capacity", [1, 4])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("owner_atoms", [32, 64])
def test_matrix_free_wide_complete_overflow_and_owner_partitions(
    capacity, splits, owner_atoms
):
    layer = model(
        parameters(257, ni=129, no=97), ni=129, no=97, width=400, device="cuda"
    )
    gen = torch.Generator().manual_seed(47)
    x = torch.randn(3, 129, generator=gen).cuda().requires_grad_()
    dy = torch.randn(3, 97, generator=gen).cuda()
    y = run(
        layer,
        x,
        atom_chunk=256,
        owner_capacity=capacity,
        owner_splits=splits,
        owner_atoms=owner_atoms,
    )
    gate((y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)), oracle(layer, x, dy))


@GPU
@pytest.mark.parametrize("capacity", [1, 4])
def test_matrix_free_csr_membership_and_overflow_are_disjoint_and_complete(capacity):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free import (
        kernels as k,
    )

    atoms, owners = 5, 5
    packed = torch.zeros(13, atoms, device="cuda")
    lows, highs = [0, 15, 0, 0, 16], [16, 17, 80, 0, 17]
    for axis in (9, 11):
        packed[axis] = torch.tensor(lows, device="cuda")
        packed[axis + 1] = torch.tensor(highs, device="cuda")
    counts = torch.zeros(owners + 1, device="cuda", dtype=torch.int32)
    offsets, cursors = torch.empty_like(counts), torch.empty_like(counts)
    ids = torch.full((capacity * atoms,), -1, device="cuda", dtype=torch.int32)
    overflow = torch.full((atoms,), -1, device="cuda", dtype=torch.int32)
    k.count_members[(1,)](packed, counts, overflow, atoms, owners, False, capacity, 256)
    k.prefix[(1,)](counts, offsets, cursors, owners, tr.next_power_of_2(owners))
    k.scatter_members[(1,)](packed, cursors, ids, atoms, False, capacity, 256)
    expected = [[] for _ in range(owners)]
    wide = []
    for atom, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        if hi <= lo:
            continue
        touched = list(range(lo // 16, (hi + 15) // 16))
        if len(touched) > capacity:
            wide.append(atom)
        else:
            for owner in touched:
                expected[owner].append(atom)
    actual_overflow = overflow[: int(counts[-1])].tolist()
    assert sorted(actual_overflow) == wide
    assert counts[:-1].tolist() == [len(entries) for entries in expected]
    for owner, entries in enumerate(expected):
        actual = ids[int(offsets[owner]) : int(offsets[owner + 1])].tolist()
        assert sorted(actual) == entries
        assert not set(actual) & set(actual_overflow)


@GPU
@pytest.mark.parametrize("preparation", ["support", "full"])
def test_matrix_free_precision_fallback_large_origin_retains_all_sites(preparation):
    layer = model(parameters(19, origin=1e6), origin=1e6, width=4, device="cuda")
    gen = torch.Generator().manual_seed(53)
    x = torch.randn(3, 65, generator=gen).cuda().requires_grad_()
    dy = torch.randn(3, 33, generator=gen).cuda()
    y = run(layer, x, preparation=preparation, atom_chunk=256, owner_capacity=1)
    gate((y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)), oracle(layer, x, dy))


@GPU
@pytest.mark.parametrize(
    "case", ["empty", "singleton", "tiny", "below", "equal", "above"]
)
def test_matrix_free_whole_product_floor_and_singleton_derivatives(case):
    center = {"empty": -10.0, "singleton": 0.0, "tiny": -0.95}.get(case, -0.5)
    # gap=.75, raw=.421875 on both singleton axes: joint norm is binary-exact.
    joint_norm = 0.421875**2
    floor = (
        joint_norm * {"below": 2.0, "equal": 1.0, "above": 0.5}.get(case, 0.0) or 1e-6
    )
    layer = model(
        torch.tensor([[0.3, 1.5, center, center]]),
        ni=2,
        no=2,
        width=1,
        floor=floor,
        device="cuda",
    )
    x = torch.tensor([[0.7, -0.2], [-0.4, 0.8]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.3, 0.9], [-0.6, 0.2]], device="cuda")
    y = run(layer, x, atom_chunk=256)
    gx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    expected = oracle(layer, x, dy)
    gate((y, gx, gp), expected)
    if case in ("tiny", "below"):
        assert expected[2][:, 2:].abs().max() > 1e-3
        assert gp[:, 2:].abs().max() > 1e-3
    elif case in ("singleton", "equal", "above"):
        assert gp[:, 2:].abs().max() < 1e-9
    else:
        assert all(not torch.count_nonzero(t) for t in (y, gx, gp))


@GPU
@pytest.mark.parametrize("atoms", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_matrix_free_empty_atoms_and_requested_gradient_branches(atoms, need_x, need_p):
    layer = model(parameters(atoms), device="cuda")
    layer.atoms.p.requires_grad_(need_p)
    x = torch.randn(3, 65, device="cuda", requires_grad=need_x)
    dy = torch.randn(3, 33, device="cuda")
    truth = oracle(layer, x, dy)
    y = run(layer, x, atom_chunk=256)
    targets = tuple(t for t, needed in ((x, need_x), (layer.atoms.p, need_p)) if needed)
    expected = (truth[0],) + tuple(
        t for t, needed in ((truth[1], need_x), (truth[2], need_p)) if needed
    )
    actual = (y, *torch.autograd.grad(y, targets, dy))
    gate(actual, expected)
    if not atoms:
        assert all(not torch.count_nonzero(t) for t in actual)


@GPU
def test_matrix_free_retained_forward_owns_parameters_scalars_and_lattice():
    layer = model(parameters(257), device="cuda")
    x = torch.randn(3, 65, device="cuda", requires_grad=True)
    dy = torch.randn(3, 33, device="cuda")
    truth = oracle(layer, x, dy)
    y = run(layer, x, atom_chunk=256)
    first = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    gate((y, *first), truth)
    with torch.no_grad():
        layer.atoms.p.add_(0.17)
        layer.kernel.amplitude_max.mul_(0.7)
        layer.kernel.sigma_max_input.mul_(0.9)
        for axis in layer.chart.axes:
            axis.start.add_(0.25)
            axis.spacing.fill_(1.1)
    newer = run(layer, x, atom_chunk=256)
    gate(
        (newer, *torch.autograd.grad(newer, (x, layer.atoms.p), dy)),
        oracle(layer, x, dy),
    )
    old = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    gate((y, *old), truth)
    # Exact CSR membership need not preserve integer-atomic insertion order;
    # dX reductions can therefore change order while retaining the snapshot.
    for value, expected in zip(old, first, strict=True):
        torch.testing.assert_close(value, expected, atol=2e-5, rtol=4e-4)


@GPU
def test_matrix_free_allocations_exclude_weight_and_global_h_g():
    from torch.utils._python_dispatch import TorchDispatchMode

    class BoundedAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))

            def inspect(value):
                if isinstance(value, torch.Tensor):
                    assert value.shape != (33, 65), "W/dW materialized"
                    assert value.shape not in ((513, 3), (3, 513), (1539,)), (
                        "global H/G materialized"
                    )
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        inspect(item)

            inspect(result)
            return result

    layer = model(parameters(513), device="cuda")
    x = torch.randn(3, 65, device="cuda", requires_grad=True)
    dy = torch.randn(3, 33, device="cuda")
    warm = run(layer, x, atom_chunk=256)
    torch.autograd.grad(warm, (x, layer.atoms.p), dy)
    retained_shapes = []

    def record_saved(value):
        retained_shapes.append(tuple(value.shape))
        return value

    with (
        torch.autograd.graph.saved_tensors_hooks(record_saved, lambda value: value),
        BoundedAllocations(),
    ):
        y = run(layer, x, atom_chunk=256)
        actual = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))
    assert (13 * 513,) in retained_shapes
    assert (513, 4) in retained_shapes
    assert (1539,) not in retained_shapes
    assert (256 * 3,) not in retained_shapes, "chunk H/G retained instead of recomputed"
    gate(actual, oracle(layer, x, dy))


@GPU
def test_matrix_free_twenty_graph_updates_match_public_moments_clock_and_live_widths():
    from benchmarks.cuda.linear.profile_product import decode
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import AtomUpdateBinding, CSTOptimizer

    layer = model(parameters(257), device="cuda")
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True, capturable=True)
    binding = AtomUpdateBinding(layer.operator)
    gen = torch.Generator().manual_seed(59)
    x = torch.randn(7, 65, generator=gen).cuda()
    dy = torch.randn(7, 33, generator=gen).cuda()
    initial_widths = decode(layer.kernel, layer.atoms.p)[:, 1].detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        y = run(layer, x, atom_chunk=256)
        (y * dy).sum().backward()
        optimizer_step(binding, opt, step_size=1e-3, polar_update="fused")
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(reference.parameters(), lr=1e-3, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    public = CSTOptimizer(base, model=reference)
    before = int(opt.state[layer.atoms.p]["step"])
    assert before == 2
    for index in range(20):
        public.zero_grad(set_to_none=True)
        expected = reference.operator.apply(x, algorithm="factored")
        (expected * dy).sum().backward()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=4e-4, atol=2e-5)
        torch.testing.assert_close(layer.atoms.p, reference.atoms.p, atol=2e-6, rtol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[reference.atoms.p][key],
                rtol=4e-4,
                atol=2e-5,
            )
        assert int(opt.state[layer.atoms.p]["step"]) == before + index + 1
        assert int(base.state[reference.atoms.p]["step"]) == before + index + 1
    assert torch.any(decode(layer.kernel, layer.atoms.p)[:, 1] != initial_widths)
    xx = x.detach().clone().requires_grad_()
    y = run(layer, xx, atom_chunk=256)
    gate((y, *torch.autograd.grad(y, (xx, layer.atoms.p), dy)), oracle(layer, xx, dy))
