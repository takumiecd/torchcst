"""Fixed-c64 widths have independent host admission and retained-backward coverage."""

from dataclasses import replace

import pytest
import torch
from test_lm_physical_radius import physical
from test_radius_envelope import make

from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    LM_PHYSICAL_DENSE_FOURIER_WEIGHT,
    LM_PHYSICAL_WIDTH_LEVELS,
    RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM,
    RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM,
    RadiusEnvelopeDenseFourierWeightLMAlgorithm,
    RadiusEnvelopeWideAmplitudeWidthLMAlgorithm,
    RadiusLMPhysicalDenseFourierWeightAlgorithm,
    RadiusLMPhysicalWidthLevelsAlgorithm,
    width_extension,
)
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs


@pytest.mark.parametrize(
    "ni,no,atoms",
    [(w, 4 * w, 128 * w) for w in (512, 1024, 2048)]
    + [(4 * w, w, 128 * w) for w in (512, 1024, 2048)],
)
@pytest.mark.parametrize("envelope", [False, True])
def test_width_metadata_and_roundtrip(ni, no, atoms, envelope):
    m = (
        make(ni, no, atoms=1, dtype=torch.float32, radius_max=50, amplitude_max=32)
        if envelope
        else physical(ni, no, 1)
    )
    c = replace(
        m.build_context(LinearInputs(torch.zeros(1, ni))),
        atom_count=atoms,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    algorithm, plan = (
        (
            RadiusEnvelopeWideAmplitudeWidthLMAlgorithm(),
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_WIDTH_LM,
        )
        if envelope
        else (RadiusLMPhysicalWidthLevelsAlgorithm(), LM_PHYSICAL_WIDTH_LEVELS)
    )
    assert algorithm.supports(c, plan.recipe).supported
    for fields in (
        {"atom_count": 8192},
        {"atom_count": atoms + 1},
        {"input_shape": (257, ni)},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
    ):
        assert not algorithm.supports(replace(c, **fields), plan.recipe).supported
    registry = make_algorithm_registry()
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize(
    "ni,no,atoms",
    [(w, 4 * w, 128 * w) for w in (512, 1024, 2048)]
    + [(4 * w, w, 128 * w) for w in (512, 1024, 2048)],
)
@pytest.mark.parametrize("rows", [7, 256])
def test_width_retained_and_multiple_forward(ni, no, atoms, rows):
    torch.manual_seed(816)
    m = physical(ni, no, atoms).cuda()
    with torch.no_grad():
        m.atoms.p[:, 0].uniform_(-0.4, 0.4)
        m.atoms.p[:, 1].uniform_(0.51, 4.0)
        m.atoms.p[:, 2].uniform_(-0.37, no + 0.23)
        m.atoms.p[:, 3].uniform_(-0.37, ni + 0.23)
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalWidthLevelsAlgorithm())
    m.selector = FixedSelector(LM_PHYSICAL_WIDTH_LEVELS, registry=registry)
    x = torch.randn(rows, ni * 2, device="cuda")[:, ::2].requires_grad_()
    y = m(x)
    later = m(x * 1.1)
    dy = torch.randn(rows, no, device="cuda")
    first = torch.autograd.grad(y, (x, m.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, m.atoms.p), dy)
    for a, b in zip(first, second):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert first[1][:, 1].abs().max() > 0
    later.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_width_cpp_does_not_admit_c8():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    m = physical(512, 2048, 8192).cuda()
    p = hierarchy_from_operator(m.execution_declaration())
    with pytest.raises(RuntimeError, match="packed pilot requires admitted"):
        width_extension().packed_levels_linear(
            torch.zeros(1, 512, device="cuda"),
            m.atoms.p,
            p.input.native_config(),
            p.output.native_config(),
            False,
            True,
        )


@pytest.mark.parametrize(
    "ni,no,atoms",
    [(4096, 1024, 196608), (8192, 2048, 393216)]
    + [(2048, 2048, coefficient * 2048) for coefficient in (32, 48, 56, 64, 72, 80, 96, 128)]
    + [(4096, 4096, coefficient * 4096) for coefficient in (32, 48, 64, 96)],
)
def test_dense_fourier_weight_is_an_explicit_projection_plan(ni, no, atoms):
    layer = make(
        ni, no, atoms=1, dtype=torch.float32, radius_max=512, amplitude_max=32
    )
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(1, ni))),
        atom_count=atoms,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    algorithm = RadiusEnvelopeDenseFourierWeightLMAlgorithm()
    plan = RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM
    assert algorithm.supports(context, plan.recipe).supported
    for changed in (
        {"atom_count": atoms + 1},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
        {"precision": replace(context.precision, allow_tf32=True)},
    ):
        assert not algorithm.supports(
            replace(context, **changed), plan.recipe
        ).supported
    registry = make_algorithm_registry()
    with pytest.raises((KeyError, ValueError)):
        registry.validate_plan(plan)
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize(
    "atom_grad,input_grad", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    "ni,no,atoms",
    [(4096, 1024, 196608), (8192, 2048, 393216), (2048, 2048, 131072), (2048, 2048, 262144)],
)
def test_dense_fourier_weight_at_tile_boundaries_and_row_tail(
    ni, no, atoms, atom_grad, input_grad, monkeypatch
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    rows = 9
    model = physical(ni, no, atoms).cuda()
    seed = torch.Generator(device="cuda").manual_seed(17)
    p = model.atoms.p
    radii = p.new_tensor([0.5, 1, 1.01, 2, 4, 8, 16, 64, 64.01, 512])
    with torch.no_grad():
        p[:, 0].uniform_(-0.04, 0.04, generator=seed)
        p[:, 1].copy_(radii[torch.arange(atoms, device=p.device) % len(radii)])
        p[:, 2].uniform_(-0.37, no + 0.23, generator=seed)
        p[:, 3].uniform_(-0.37, ni + 0.23, generator=seed)
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalWidthLevelsAlgorithm())
    registry.register(RadiusLMPhysicalDenseFourierWeightAlgorithm())
    x = torch.randn(rows, ni, device=p.device, generator=seed).requires_grad_(
        input_grad
    )
    p.requires_grad_(atom_grad)
    inputs = tuple(t for t in (x, p) if t.requires_grad)
    dy = torch.randn(rows, no, device=p.device, generator=seed)
    model.selector = FixedSelector(LM_PHYSICAL_WIDTH_LEVELS, registry=registry)
    expected = model(x)
    expected_grads = torch.autograd.grad(expected, inputs, dy)
    model.selector = FixedSelector(LM_PHYSICAL_DENSE_FOURIER_WEIGHT, registry=registry)
    actual = model(x)
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=4e-3)
        assert (a - b).abs().max() <= 3e-4
