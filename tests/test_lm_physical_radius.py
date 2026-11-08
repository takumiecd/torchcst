"""The physical LM path must retain the radius derivative omitted by Polar."""

import copy
from dataclasses import replace

import pytest
import torch

from torchcst import (
    ChebyshevSpec,
    CSTLinear,
    FixedSelector,
    chart_presets,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
    LM_LEVELS,
    LM_PHYSICAL_LEVELS,
    RadiusLMLevelsAlgorithm,
    RadiusLMPhysicalLevelsAlgorithm,
)
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs


def physical(ni, no, atoms):
    return CSTLinear(
        chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
        atoms=atoms,
        dtype=torch.float32,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.physical(radius_init=(0.5, 4.0), radius_floor=0.5),
        ),
    )


@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
def test_physical_admission_is_separate_from_polar(ni, no):
    model = physical(ni, no, 1)
    c = model.build_context(LinearInputs(torch.zeros(1, ni)))
    c = replace(
        c,
        atom_count=8192,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    assert (
        RadiusLMPhysicalLevelsAlgorithm()
        .supports(c, LM_PHYSICAL_LEVELS.recipe)
        .supported
    )
    assert not RadiusLMLevelsAlgorithm().supports(c, LM_LEVELS.recipe).supported
    for fields in [
        {"atom_count": 8191},
        {"input_shape": (257, ni)},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
    ]:
        assert (
            not RadiusLMPhysicalLevelsAlgorithm()
            .supports(replace(c, **fields), LM_PHYSICAL_LEVELS.recipe)
            .supported
        )
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalLevelsAlgorithm())
    assert (
        registry.loads_plan(registry.dumps_plan(LM_PHYSICAL_LEVELS))
        == LM_PHYSICAL_LEVELS
    )
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(LM_PHYSICAL_LEVELS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
@pytest.mark.parametrize("rows", [7, 65, 256])
def test_full_physical_radius_vjp_and_retained_backward(ni, no, rows):
    torch.manual_seed(814)
    base = physical(ni, no, 8192)
    with torch.no_grad():
        base.atoms.p[:, 0].uniform_(-0.4, 0.4)
        base.atoms.p[:, 1].uniform_(0.51, 4.0)
        base.atoms.p[:, 2].uniform_(-0.37, no + 0.23)
        base.atoms.p[:, 3].uniform_(-0.37, ni + 0.23)
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalLevelsAlgorithm())
    raw = torch.randn(rows, ni * 2, device="cuda")
    dy = torch.randn(rows, no, device="cuda")
    results = []
    for plan in [None, LM_PHYSICAL_LEVELS]:
        layer = (
            copy.deepcopy(base).double() if plan is None else copy.deepcopy(base).cuda()
        )
        if plan is not None:
            layer.selector = FixedSelector(plan, registry=registry)
        x = (
            (raw.cpu().double() if plan is None else raw)[:, ::2]
            .detach()
            .requires_grad_()
        )
        cotangent = dy.cpu().double() if plan is None else dy
        y = layer(x)
        gradients = torch.autograd.grad(
            y, (x, layer.atoms.p), cotangent, retain_graph=True
        )
        again = torch.autograd.grad(y, (x, layer.atoms.p), cotangent)
        for a, b in zip(gradients, again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert gradients[1][:, 1].abs().max() > 0
        results.append((y, *gradients))
    for a, b in zip(*results):
        torch.testing.assert_close(
            a.cpu().double(), b.cpu().double(), atol=3e-5, rtol=3e-4
        )
