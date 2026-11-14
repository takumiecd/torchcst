"""Separate cohort admission and retained complete M8 gradients at A16384."""

from dataclasses import replace

import pytest
import torch
from test_lm_physical_radius import physical
from test_radius_envelope import make

from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_c16 import (
    LM_PHYSICAL_C16_LEVELS,
    RADIUS_ENVELOPE_WIDE_AMPLITUDE_C16_LM,
    RadiusEnvelopeWideAmplitudeC16LMAlgorithm,
    RadiusLMPhysicalC16LevelsAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_multirow import (
    LM_PHYSICAL_LEVELS,
    RadiusLMPhysicalLevelsAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
    RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM,
    RadiusEnvelopeWideAmplitudeLMAlgorithm,
)
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs


@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
@pytest.mark.parametrize("envelope", [False, True])
def test_c16_admission_preserves_old_cohort_and_roundtrip(ni, no, envelope):
    m = (
        make(ni, no, atoms=1, dtype=torch.float32, radius_max=50, amplitude_max=32)
        if envelope
        else physical(ni, no, 1)
    )
    c = m.build_context(LinearInputs(torch.zeros(1, ni)))
    c = replace(
        c,
        atom_count=16384,
        input_shape=(256, ni),
        input_strides=(ni, 1),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    old, old_plan, new, new_plan = (
        (
            RadiusEnvelopeWideAmplitudeLMAlgorithm(),
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM,
            RadiusEnvelopeWideAmplitudeC16LMAlgorithm(),
            RADIUS_ENVELOPE_WIDE_AMPLITUDE_C16_LM,
        )
        if envelope
        else (
            RadiusLMPhysicalLevelsAlgorithm(),
            LM_PHYSICAL_LEVELS,
            RadiusLMPhysicalC16LevelsAlgorithm(),
            LM_PHYSICAL_C16_LEVELS,
        )
    )
    assert new.supports(c, new_plan.recipe).supported
    assert not old.supports(c, old_plan.recipe).supported
    assert old.supports(replace(c, atom_count=8192), old_plan.recipe).supported
    for fields in (
        {"atom_count": 8192},
        {"atom_count": 16383},
        {"input_shape": (257, ni)},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
    ):
        assert not new.supports(replace(c, **fields), new_plan.recipe).supported
    registry = make_algorithm_registry()
    registry.register(new)
    assert registry.loads_plan(registry.dumps_plan(new_plan)) == new_plan
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(new_plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
@pytest.mark.parametrize("rows", [1, 7, 128, 256])
def test_c16_outstanding_and_retained_gradients(ni, no, rows):
    torch.manual_seed(816)
    m = physical(ni, no, 16384).cuda()
    with torch.no_grad():
        m.atoms.p[:, 0].uniform_(-0.4, 0.4)
        m.atoms.p[:, 1].uniform_(0.51, 4.0)
        m.atoms.p[:, 2].uniform_(-0.37, no + 0.23)
        m.atoms.p[:, 3].uniform_(-0.37, ni + 0.23)
    registry = make_algorithm_registry()
    registry.register(RadiusLMPhysicalC16LevelsAlgorithm())
    m.selector = FixedSelector(LM_PHYSICAL_C16_LEVELS, registry=registry)
    x = torch.randn(rows, ni * 2, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(rows, no, device="cuda")
    y = m(x)
    later = m(x * 1.1)
    first = torch.autograd.grad(y, (x, m.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, m.atoms.p), dy)
    for a, b in zip(first, second):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert first[1][:, 1].abs().max() > 0
    later.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("ni,no", [(512, 2048), (2048, 512)])
def test_native_builds_keep_distinct_atom_admission(ni, no):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
        _extension,
    )
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_c16 import (
        c16_extension,
    )

    m = physical(ni, no, 16384).cuda()
    plan = hierarchy_from_operator(m.execution_declaration())
    config = (plan.input.native_config(), plan.output.native_config())
    x = torch.zeros(1, ni, device="cuda")
    with pytest.raises(RuntimeError, match="packed pilot requires admitted"):
        _extension(8).packed_levels_linear(x, m.atoms.p, *config, False, True)
    with pytest.raises(RuntimeError, match="packed pilot requires admitted"):
        c16_extension().packed_levels_linear(
            x, m.atoms.p[:8192].contiguous(), *config, False, True
        )
