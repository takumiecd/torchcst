"""Same-precision replay and public callback guards for the opt-in FP32 Plan."""

import os

import pytest
import torch
from test_full_step_graph import (  # noqa: F401 -- collect the identical contract cases
    cohort,
    test_current_boundary_wide_atoms_gradients_and_version,
    test_public_radius_entrance_still_rejects_capture,
    test_replay_invalidates_before_changed_contract,
)

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires L4 and frozen cohort inputs",
)


@pytest.fixture(autouse=True)
def choose_fp32_plan(monkeypatch):
    monkeypatch.setattr(
        plans, "GLOBAL_OWNER_WINDOW_RECOMPUTE", plans.GLOBAL_OWNER_WINDOW_FP32
    )
