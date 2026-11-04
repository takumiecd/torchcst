"""Existing whole-step callback and evaluation contracts on known CTA banks."""

import os

import pytest
import torch
from test_fp32_full_step_graph import (  # noqa: F401
    test_capture_restores_counted_metrics_and_numerical_state,
    test_evaluation_roundtrip_renews_both_certificates_without_update,
)
from test_full_step_graph import (  # noqa: F401
    cohort,
    test_current_boundary_wide_atoms_gradients_and_version,
    test_public_radius_entrance_still_rejects_capture,
    test_replay_invalidates_before_changed_contract,
)

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires L4 and frozen cohort",
)


@pytest.fixture(autouse=True)
def choose_known_bank(monkeypatch):
    monkeypatch.setattr(
        plans,
        "GLOBAL_OWNER_WINDOW_RECOMPUTE",
        plans.GLOBAL_OWNER_WINDOW_FP32_KNOWN_BANK,
    )


def test_known_bank_public_capture_rejected(request):
    m, _, tx, _, _, p, _ = request.getfixturevalue("cohort")
    with (
        torch.cuda.graph(torch.cuda.CUDAGraph(), stream=torch.cuda.current_stream()),
        pytest.raises(RuntimeError, match="not CUDA Graphs"),
    ):
        p.extension.research_owner_window_fp32_known_bank(
            tx.new_zeros((4, 1024)), m[2].atoms.p, *p.configs, False
        )
