"""FP32 retained tensors, recomputation, row-parallel gradients, unchanged scope."""

import copy
from dataclasses import replace

import pytest
import torch
from test_ffn_rectangular_2048 import context

from torchcst import (
    ChebyshevSpec,
    CSTLinear,
    FixedSelector,
    RadiusBounds,
    chart_presets,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.packed import (
    PACKED_NATIVE,
    PACKED_PARALLEL,
    PACKED_RECOMPUTE,
    RadiusPackedAlgorithm,
    RadiusPackedParallelAlgorithm,
    RadiusPackedRecomputeAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.row8 import (
    ROW8_NATIVE,
    RadiusRow8Algorithm,
)

CASES = [
    (RadiusPackedAlgorithm, PACKED_NATIVE),
    (RadiusPackedRecomputeAlgorithm, PACKED_RECOMPUTE),
    (RadiusPackedParallelAlgorithm, PACKED_PARALLEL),
]


def fixture(ni, no):
    model = CSTLinear(
        chart_presets.regular_grid(no, spacing=1, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1, geometry="flat_torus"),
        atoms=4096,
        dtype=torch.float32,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.polar(
                amplitude_max=1,
                w_c=0.3,
                input_bounds=RadiusBounds(
                    minimum=0.5, birth=4, maximum=50, upper_floor=1
                ),
            ),
        ),
    )
    with torch.no_grad():
        ratios = torch.linspace(-0.999, 0.999, 4096)
        ratios[:4] = torch.tensor([0, 1e-7, -1e-7, 0.3])
        rho = (1 + (torch.arange(4096) % 4)).sqrt()
        model.atoms.p[:, 0] = ratios * rho
        model.atoms.p[:, 1] = (1 - ratios.square()).sqrt() * rho
        model.atoms.p[:, 2] = torch.linspace(-0.5, no + 0.5, 4096)
        model.atoms.p[:, 3] = torch.linspace(ni + 0.5, -0.5, 4096)
    return model


def registry():
    r = make_algorithm_registry()
    r.register(RadiusRow8Algorithm())
    for cls, plan in CASES:
        r.register(cls())
    return r


@pytest.mark.parametrize("cls,plan", CASES)
def test_metadata_roundtrip_retention_and_no_default(cls, plan):
    ctx = context(256, 1024, count=4096)
    a = cls()
    assert a.supports(ctx, plan.recipe).supported
    r = registry()
    assert r.loads_plan(r.dumps_plan(plan)) == plan
    assert not a.supports(replace(ctx, input_shape=(257, 256)), plan.recipe).supported
    assert not a.supports(
        replace(ctx, execution_mode="cuda_graph"), plan.recipe
    ).supported
    assert not a.supports(replace(ctx, dtype=torch.float64), plan.recipe).supported
    with pytest.raises(ValueError):
        a.validate_recipe(replace(plan.recipe, save_response=a.recompute))
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(plan)
    # Construct GPU fixtures on CPU as well, so skipped CUDA tests cannot hide API mistakes.
    assert fixture(256, 1024).atom_count == 4096


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@GPU
@pytest.mark.parametrize("cls,plan", CASES)
@pytest.mark.parametrize("ni,no", [(256, 1024), (1024, 256)])
@pytest.mark.parametrize("rows", [7, 8, 9, 255, 256])
def test_cuda_all_gradients_storage_tails_retained_and_version(cls, plan, ni, no, rows):
    torch.manual_seed(171)
    base = fixture(ni, no)
    r = registry()
    models = [copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()]
    for m, p in zip(models, [ROW8_NATIVE, plan]):
        m.selector = FixedSelector(p, registry=r)
    x = torch.randn(rows, ni, device="cuda")
    dy = torch.randn(rows, no, device="cuda")
    outputs = []
    saved = []
    for index, m in enumerate(models):
        xx = x.clone().requires_grad_()

        def pack(t, index=index):
            if index == 1:
                saved.append((tuple(t.shape), t.dtype))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = m(xx)
        gx, gp = torch.autograd.grad(y, (xx, m.atoms.p), dy, retain_graph=True)
        again = torch.autograd.grad(y, (xx, m.atoms.p), dy, retain_graph=True)
        for a, b in zip((gx, gp), again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        outputs.append((y, gx, gp))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    assert all(dtype != torch.float64 for shape, dtype in saved)
    retained = any(shape == (rows, 4096) for shape, dtype in saved)
    assert retained is (not cls().recompute)
    # Hooks intentionally delegate saved-version policy to hooks. Test normal
    # SavedVariables/ReadLease separately, without a custom pack hook.
    yy = models[1](xx)
    with torch.no_grad():
        models[1].atoms.p.add_(0.0001)
    with pytest.raises(
        RuntimeError, match="modified (by an inplace operation|inplace)"
    ):
        torch.autograd.grad(yy, (xx, models[1].atoms.p), dy)


@GPU
@pytest.mark.parametrize("cls,plan", CASES)
@pytest.mark.parametrize("freeze", ["input", "atoms"])
def test_cuda_one_gradient_branch(cls, plan, freeze):
    torch.manual_seed(172)
    base = fixture(256, 1024)
    r = registry()
    outputs = []
    x = torch.randn(9, 256, device="cuda")
    dy = torch.randn(9, 1024, device="cuda")
    for selected in [ROW8_NATIVE, plan]:
        model = copy.deepcopy(base).cuda()
        model.atoms.p.requires_grad_(freeze != "atoms")
        model.selector = FixedSelector(selected, registry=r)
        xx = x.clone().requires_grad_(freeze != "input")
        y = model(xx)
        target = xx if freeze == "atoms" else model.atoms.p
        outputs.append((y, torch.autograd.grad(y, target, dy)[0]))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
