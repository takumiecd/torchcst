"""Reverse owner contraction: independent VJPs and byte-exact Adam state."""

import pytest
import test_radius_compact_keyed_owner as keyed
import test_radius_owner_response_derivatives as derivative
import torch
from test_radius_cell_mass_cuda import GPU

from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_COMPACT_FUSED as FUSED,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_COMPACT_KEYED as BASE,
)


def test_fused_declaration_roundtrip():
    RadiusCellMassAlgorithm().validate_recipe(FUSED.recipe)
    registry = make_algorithm_registry()
    assert registry.loads_plan(registry.dumps_plan(FUSED)) == FUSED


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_fused_independent_vjp(monkeypatch, dtype, batch, needs_x, needs_p, skew):
    monkeypatch.setattr(derivative, "NEW", FUSED)
    derivative.test_all_vjps_boundary_images_skew_retained_and_offload(
        dtype, batch, needs_x, needs_p, skew
    )


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_fused_exact_state(monkeypatch, dtype, batch, needs_x, needs_p, skew):
    monkeypatch.setattr(keyed, "OLD", BASE)
    monkeypatch.setattr(keyed, "COMPACT", FUSED)
    keyed.test_mode9_vs10_mandatory_exact_y_dx_dp_parameter_adam_bytes(
        dtype, batch, needs_x, needs_p, skew
    )
