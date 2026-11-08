"""LM projection admission and all-gradient comparison to the existing native route."""

import copy
from dataclasses import replace

import pytest
import torch
from test_ffn_rectangular_2048 import context
from test_variance_initialization import layer

from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
    LM_LEVELS,
    LM_LEVELS_STREAM64,
    RadiusLMLevelsAlgorithm,
    RadiusLMLevelsStream64Algorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
    GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048,
)

CASES = [
    (RadiusLMLevelsAlgorithm, LM_LEVELS),
    (RadiusLMLevelsStream64Algorithm, LM_LEVELS_STREAM64),
]
SHAPES = [
    (256, 256, 2048),
    (256, 1024, 4096),
    (1024, 256, 4096),
    (512, 512, 4096),
    (512, 2048, 8192),
    (2048, 512, 8192),
]


def registry():
    r = make_algorithm_registry()
    for cls, _ in CASES:
        r.register(cls())
    return r


@pytest.mark.parametrize("cls,plan", CASES)
@pytest.mark.parametrize("ni,no,atoms", SHAPES)
def test_explicit_lm_scope(cls, plan, ni, no, atoms):
    c = context(ni, no, count=atoms)
    assert cls().supports(c, plan.recipe).supported
    assert registry().loads_plan(registry().dumps_plan(plan)) == plan
    for changes in [
        {"atom_count": atoms + 1},
        {"input_shape": (257, ni)},
        {"execution_mode": "cuda_graph"},
        {"dtype": torch.float64},
    ]:
        assert not cls().supports(replace(c, **changes), plan.recipe).supported
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("cls,plan", CASES)
@pytest.mark.parametrize("ni,no,atoms", SHAPES)
@pytest.mark.parametrize("rows", [7, 65])
def test_cuda_lm_all_vjps(cls, plan, ni, no, atoms, rows):
    torch.manual_seed(813)
    base = layer(ni, no, atoms=atoms, dtype=torch.float32)
    # Includes wrapped boundaries, exactly zero amplitude and broad/narrow atoms.
    with torch.no_grad():
        amp = torch.linspace(-0.8, 0.8, atoms)
        base.atoms.p[:, 0] = amp
        base.atoms.p[:, 1] = (1 - amp.square()).sqrt()
        base.atoms.p[0, :2] = torch.tensor([0.0, 1.0])
        base.atoms.p[:, 2] = torch.linspace(-0.5, no + 0.5, atoms)
        base.atoms.p[:, 3] = torch.linspace(ni + 0.5, -0.5, atoms)
    baseline = (
        GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048
        if max(ni, no) > 1024
        else GLOBAL_ACTIVE_GRAD4_NATIVE
    )
    raw = torch.randn(rows, ni * 2, device="cuda")
    dy = torch.randn(rows, no, device="cuda")
    outputs = []
    for selected in [baseline, plan]:
        m = copy.deepcopy(base).cuda()
        m.selector = FixedSelector(selected, registry=registry())
        x = raw[:, ::2].detach().requires_grad_()
        y = m(x)
        grads = torch.autograd.grad(y, (x, m.atoms.p), dy, retain_graph=True)
        repeated = torch.autograd.grad(y, (x, m.atoms.p), dy)
        for a, b in zip(grads, repeated):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        outputs.append((y, *grads))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
