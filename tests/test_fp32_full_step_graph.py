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


def test_public_fp32_entrance_itself_rejects_capture(request):
    model, _, tx, _, _, program, _ = request.getfixturevalue("cohort")
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph, stream=torch.cuda.current_stream()),
        pytest.raises(RuntimeError, match="not CUDA Graphs"),
    ):
        program.extension.research_owner_window_fp32(
            tx.new_zeros((4, 1024)), model[2].atoms.p, *program.configs, False
        )


def test_capture_restores_counted_metrics_and_numerical_state(request):
    from benchmarks.training.full_step_graph import exact, snapshot

    model, opt, _, _, indices, program, _ = request.getfixturevalue("cohort")
    program.totals[0].fill_(12.5)
    program.totals[1].fill_(7)
    wanted = snapshot(model, opt)
    program.indices.copy_(indices)
    program.capture(preserve_totals=True)
    assert float(program.totals[0]) == 12.5
    assert int(program.totals[1]) == 7
    assert exact(wanted, snapshot(model, opt))["all_byte_equal"]


def test_evaluation_roundtrip_renews_both_certificates_without_update(request):
    from benchmarks.training.full_step_graph import PreparedStep, exact, snapshot
    from benchmarks.training.full_step_graph_learning import renew_certificates

    model, opt, tx, ty, _, program, _ = request.getfixturevalue("cohort")
    wanted = snapshot(model, opt)
    model.eval()
    with torch.no_grad():
        model(tx[:4])
    model.train()
    assert not program.eligible()
    renew_certificates(model, opt, tx[:4], "cst")
    assert exact(wanted, snapshot(model, opt))["all_byte_equal"]
    renewed = PreparedStep(model, opt, tx, ty, "cst")
    renewed.capture(preserve_totals=True)
    assert renewed.eligible()
    assert exact(wanted, snapshot(model, opt))["all_byte_equal"]
